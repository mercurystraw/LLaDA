'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import json
import torch
import re
import time
from pathlib import Path
import random
from contextlib import nullcontext
from datetime import timedelta
import jinja2
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from generate import generate, var_generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("illada_dist")
class ILLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=5,
        max_length=None,
        batch_size=32,
        mc_num=128,
        llh='mc',
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        temperature=0.,
        remasking='low_confidence',
        var=False,
        add_bos_token=False,
        padd_eos=False,
        end_think_text='</think>',
        end_think_logit_boost=0.,
        end_think_boost_power=2.,
        throughput_output=None,
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: iLLaDA model path.
            mask_id: The token id of [MASK] is 5.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            llh: Likelihood method. `mc` or `confidence`.
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator(
            kwargs_handlers=[
                accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=16))
            ]
        )
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        load_context = (
            self.accelerator.main_process_first()
            if self.accelerator is not None else nullcontext()
        )
        with load_context:
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, **model_kwargs
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id

        self.mc_num = mc_num
        self.llh = llh
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        model_config = getattr(getattr(self.model, 'module', self.model), 'config', None)
        self.max_length = int(
            max_length or getattr(model_config, 'max_position_embeddings', 4096)
        )
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.remasking = remasking
        self.var = var
        self.add_bos_token = add_bos_token
        self.padd_eos = padd_eos
        self.end_think_token_ids = self.tokenizer.encode(
            end_think_text, add_special_tokens=False
        )
        self.end_think_logit_boost = end_think_logit_boost
        self.end_think_boost_power = end_think_boost_power
        self.throughput_output = throughput_output

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace('/', '__')

    @staticmethod
    def _merge_system_into_user(chat_history):
        system_messages = [
            str(message['content']).strip()
            for message in chat_history if message['role'] == 'system'
        ]
        messages = [
            {'role': message['role'], 'content': str(message['content'])}
            for message in chat_history if message['role'] != 'system'
        ]
        if system_messages:
            system_text = '\n\n'.join(system_messages)
            if messages and messages[0]['role'] == 'user':
                messages[0]['content'] = f"{system_text}\n\n{messages[0]['content']}"
            else:
                messages.insert(0, {'role': 'user', 'content': system_text})
        return messages

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        def render(messages):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    continue_final_message=not add_generation_prompt,
                    **kwargs,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **kwargs)

        try:
            return render(chat_history)
        except jinja2.exceptions.TemplateError as exc:
            if 'System role not supported' not in str(exc):
                raise
            return render(self._merge_system_into_user(chat_history))

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        if self.padd_eos:
            eos = torch.full(
                (batch.shape[0], 1), self.tokenizer.eos_token_id,
                dtype=batch.dtype, device=batch.device
            )
            model_input = torch.cat([batch, eos], dim=-1)
        else:
            model_input = batch

        logits = self.model(model_input).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood_mc(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def get_loglikelihood_confidence(self, prefix, target):
        clean_seq = torch.concatenate([prefix, target])[None, :].to(self.device)
        noisy_seq = torch.concatenate([
            prefix, torch.full_like(target, self.mask_id)
        ])[None, :].to(self.device)
        prompt_index = torch.arange(noisy_seq.shape[1], device=self.device) < len(prefix)

        losses = []
        for _ in range(len(target)):
            mask_indices = noisy_seq == self.mask_id
            logits = self.get_logits(noisy_seq, prompt_index)
            loss = F.cross_entropy(
                logits[mask_indices], clean_seq[mask_indices], reduction='none'
            )
            min_loss, min_index = torch.min(loss, dim=-1)
            losses.append(min_loss.item())

            transfer = torch.full_like(clean_seq[mask_indices], self.mask_id)
            transfer[min_index] = clean_seq[mask_indices][min_index]
            noisy_seq[mask_indices] = transfer

        return -sum(losses)

    def get_loglikelihood(self, prefix, target):
        if self.llh == 'mc':
            return self.get_loglikelihood_mc(prefix, target)
        if self.llh == 'confidence':
            return self.get_loglikelihood_confidence(prefix, target)
        raise ValueError(f'Unknown likelihood method: {self.llh}')

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= self.max_length

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        output_tokens = 0
        generated_compute_tokens = 0
        generation_seconds = 0.
        throughput_output_path = None
        stream_file = None
        partial_throughput_path = None
        if self.throughput_output:
            throughput_output_path = Path(self.throughput_output)
            throughput_output_path.parent.mkdir(parents=True, exist_ok=True)
            stream_path = throughput_output_path.parent / f'generations.rank{self.rank}.jsonl'
            partial_throughput_path = (
                throughput_output_path.parent
                / f'throughput.rank{self.rank}.partial.json'
            )
            stream_file = stream_path.open('w', encoding='utf-8', buffering=1)

        for sample_index, elem in enumerate(tqdm(ds, desc="Generating...")):
            # iLLaDA currently evaluates one unpadded prompt at a time.
            prompt = elem["question"].unsqueeze(0)
            if self.add_bos_token:
                bos = torch.tensor([[self.tokenizer.bos_token_id]], dtype=prompt.dtype)
                prompt = torch.cat([bos, prompt], dim=1)
            prompt = prompt.to(self.device)
            available_length = self.max_length - prompt.shape[1]
            gen_length = self.gen_length or available_length
            gen_length = min(gen_length, available_length)
            gen_length = gen_length // self.block_length * self.block_length
            if gen_length <= 0:
                raise ValueError('Prompt is too long to generate one complete block.')

            stop_tokens = list(elem["until"])
            if self.tokenizer.eos_token and self.tokenizer.eos_token not in stop_tokens:
                stop_tokens.append(self.tokenizer.eos_token)

            generation_kwargs = dict(
                steps=self.steps,
                gen_length=gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
                end_think_token_ids=self.end_think_token_ids,
                end_think_logit_boost=self.end_think_logit_boost,
                end_think_boost_power=self.end_think_boost_power,
            )
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            generation_start = time.perf_counter()
            if self.var:
                generated_answer = var_generate(
                    self.model, self.tokenizer, prompt,
                    stop_tokens=stop_tokens, **generation_kwargs
                )
            else:
                generated_answer = generate(
                    self.model, prompt, **generation_kwargs
                )
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            sample_generation_seconds = time.perf_counter() - generation_start
            sample_compute_tokens = generated_answer.shape[1] - prompt.shape[1]
            generation_seconds += sample_generation_seconds
            generated_compute_tokens += sample_compute_tokens
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]
            sample_output_tokens = len(self.tokenizer.encode(
                generated_answer, add_special_tokens=False
            ))
            output_tokens += sample_output_tokens

            out.append(generated_answer)

            if stream_file is not None and partial_throughput_path is not None:
                request = requests[sample_index]
                stream_file.write(json.dumps({
                    'rank': self.rank,
                    'sample_index': sample_index,
                    'task_name': request.task_name,
                    'doc_id': request.doc_id,
                    'prompt': request.args[0],
                    'response': generated_answer,
                    'output_tokens': sample_output_tokens,
                    'generated_compute_tokens': sample_compute_tokens,
                    'generation_seconds': sample_generation_seconds,
                }, ensure_ascii=False) + '\n')
                stream_file.flush()

                partial_stats = {
                    'rank': self.rank,
                    'completed_samples': sample_index + 1,
                    'output_tokens': output_tokens,
                    'generated_compute_tokens': generated_compute_tokens,
                    'generation_seconds': generation_seconds,
                    'tokens_per_second': output_tokens / generation_seconds,
                    'compute_tokens_per_second': (
                        generated_compute_tokens / generation_seconds
                    ),
                }
                partial_tmp_path = partial_throughput_path.with_suffix('.json.tmp')
                partial_tmp_path.write_text(
                    json.dumps(partial_stats, indent=2) + '\n', encoding='utf-8'
                )
                partial_tmp_path.replace(partial_throughput_path)

        if stream_file is not None:
            stream_file.close()

        local_stats = torch.tensor(
            [len(out), output_tokens, generated_compute_tokens, generation_seconds],
            dtype=torch.float64,
            device=self.device,
        )
        if self.accelerator is not None:
            all_stats = self.accelerator.gather(local_stats).reshape(-1, 4)
        else:
            all_stats = local_stats.unsqueeze(0)

        if self.rank == 0:
            total_samples = int(all_stats[:, 0].sum().item())
            total_output_tokens = int(all_stats[:, 1].sum().item())
            total_compute_tokens = int(all_stats[:, 2].sum().item())
            summed_gpu_seconds = float(all_stats[:, 3].sum().item())
            wall_seconds = float(all_stats[:, 3].max().item())
            stats = {
                'num_processes': self.world_size,
                'num_samples': total_samples,
                'output_tokens': total_output_tokens,
                'generated_compute_tokens': total_compute_tokens,
                'generation_seconds': wall_seconds,
                'tokens_per_second': total_output_tokens / wall_seconds,
                'tokens_per_second_per_gpu': total_output_tokens / summed_gpu_seconds,
                'compute_tokens_per_second': total_compute_tokens / wall_seconds,
                'mean_output_tokens_per_sample': total_output_tokens / total_samples,
            }
            print(
                f"Generation throughput: {stats['tokens_per_second']:.3f} tokens/s "
                f"({total_output_tokens} output tokens, {total_compute_tokens} computed "
                f"tokens in {wall_seconds:.3f}s, "
                f"{self.world_size} process(es))"
            )
            if throughput_output_path is not None:
                throughput_output_path.write_text(
                    json.dumps(stats, indent=2) + '\n', encoding='utf-8'
                )

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
