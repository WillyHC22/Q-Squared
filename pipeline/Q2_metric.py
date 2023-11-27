import re
import os 
import json
import torch
import spacy
import string
import logging
import pathlib  
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_score import score
from collections import Counter
import allennlp_models.pair_classification
from allennlp.predictors.predictor import Predictor
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForQuestionAnswering


class score_utils():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
        return re.sub(' +', ' ', text).strip()


    def f1_score(self, a_gold, a_pred):
        if a_pred == '':
            return 0
        gold_toks = self.clean_text(a_gold).split()
        pred_toks = self.clean_text(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def f1_bert_score(self, a_gold, a_pred):
        P, R, F1 = score(a_pred, a_gold, lang="en", verbose=True)
        return F1.mean().item()




class QG_Setup(score_utils):
    def __init__(self):
        super().__init__()
        self.qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qg_model.to(self.device)

    def get_answer_candidates(self, text):
        doc = self.nlp(text)
        candidates = [ent.text for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        candidates = [cand for cand in candidates if cand.lower() != 'i']
        return candidates

    def get_question_greedy(self, answer, context, max_length=128):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors='pt').to(self.device)

        output = self.qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                max_length=max_length)

        question = self.qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
        return question

    def get_questions_beam(self, answer, context, max_length=128, beam_size=5, num_return=5):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors='pt').to(self.device)
        beam_outputs = self.qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                        max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                        num_return_sequences=num_return, early_stopping=True)

        for beam_output in beam_outputs:
            all_questions.append(self.qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions


    def get_questions_sample(self, answer, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors='pt').to(self.device)

        sampled_outputs = self.qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                            max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                            num_return_sequences=num_return)

        for sampled in sampled_outputs:
            all_questions.append(self.qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions


class QA_Setup():
    def __init__(self):
        self.qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.qa_model.to(self.device)

    def get_answer(self, question, text):  # Code taken from https://huggingface.co/transformers/task_summary.html
        inputs = self.qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.qa_model(**inputs, return_dict=False)

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        ans = self.qa_tokenizer.convert_tokens_to_string(self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return ans



class Q2_Scorer(score_utils):
    def __init__(self):
        super().__init__()
        self.INVALID_QUESTION = -1
        self.NO_ANS = '[CLS]'
        self.NO_VALID_QUESTIONS = 'NO_Q'
        self.qg_setup = QG_Setup()
        self.qa_setup = QA_Setup()


    def filter_questions(self, exp_ans, pred_ans):
        if pred_ans == self.NO_ANS:
            return 'NO MATCH'
        if self.clean_text(exp_ans) != self.clean_text(pred_ans):
            return 'NO MATCH'
        return 'VALID'


    def non_personal(self, question):
        question_tok = self.nlp(question)
        for tok in question_tok:
            if tok.dep_ == 'nsubj':
                if tok.text.lower() == 'i' or tok.text.lower() == 'you':
                    return False
            elif tok.dep_ == 'poss':
                if tok.text.lower() == 'my' or tok.text.lower() == 'your':
                    return False
        return True


    def single_question_score(self, question, cand, response, knowledge):
        pred_ans = self.qa_setup.get_answer(question, response)

        if self.filter_questions(cand, pred_ans) == 'VALID':
            knowledge_ans = self.qa_setup.get_answer(question, knowledge)
            if knowledge_ans != self.NO_ANS:
                return self.f1_score(cand, knowledge_ans), knowledge_ans
            else:
                return 0, self.NO_ANS
        else:
            return self.INVALID_QUESTION, self.INVALID_QUESTION


    def get_response_score(self, response, knowledge, gen_method, single, remove_personal):
        f1 = 0
        num_questions = 0

        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []

        candidates = self.qg_setup.get_answer_candidates(response)
        for cand in candidates:
            if gen_method == 'greedy':
                questions = [self.qg_setup.get_question_greedy(cand, response)]
            elif gen_method == 'beam':
                questions = self.qg_setup.get_questions_beam(cand, response)
            else:
                questions = self.qg_setup.get_questions_sample(cand, response)

            for question in questions:
                if not remove_personal or self.non_personal(question):
                    question_score, knowledge_ans = self.single_question_score(question, cand, response, knowledge)
                    if question_score != self.INVALID_QUESTION:
                        num_questions += 1
                        f1 += question_score

                        valid_questions.append(question)
                        valid_cands.append(cand)
                        knowledge_answers.append(knowledge_ans)
                        scores.append(question_score)

                        if single:
                            break
        if num_questions:
            avg_f1 = f1 / num_questions
        else:
            avg_f1 = self.INVALID_QUESTION
        return avg_f1, valid_questions, valid_cands, knowledge_answers, scores


    def response_questions_stats(self, response, knowledge, gen_method, single, remove_personal):
        num_questions = 0
        num_no_ans = 0

        candidates = self.qg_setup.get_answer_candidates(response)
        for cand in candidates:
            if gen_method == 'greedy':
                questions = [self.qg_setup.get_question_greedy(cand, response)]
            elif gen_method == 'beam':
                questions = self.qg_setup.get_questions_beam(cand, response)
            else:
                questions = self.qg_setup.get_questions_sample(cand, response)

            for question in questions:
                if not remove_personal or self.non_personal(question):
                    pred_ans = self.qa_setup.get_answer(question, response)

                    if self.filter_questions(cand, pred_ans) == 'VALID':
                        num_questions += 1
                        knowledge_ans = self.qa_setup.get_answer(question, knowledge)
                        if knowledge_ans == self.NO_ANS:
                            num_no_ans += 1
                        if single:
                            break
        return num_questions, num_no_ans


    def get_stats(self, in_path, gen_method, single, remove_personal):
        num_questions = 0
        num_no_ans = 0
        df = pd.read_csv(in_path)
        for _, row in df.iterrows():
            q, no_ans = self.response_questions_stats(row['response'], row['knowledge'], gen_method, single, remove_personal)
            num_questions += q
            num_no_ans += no_ans

        print("Total valid questions: {0}".format(num_questions))
        print("No answer: {0}".format(num_no_ans / num_questions))


    def calc_scores(self, in_path, gen_method, single, remove_personal, out_path='', save_steps=False):
        print(in_path, gen_method, single, remove_personal)
        print(save_steps, flush=True)
        q_scores = []
        df = pd.read_csv(in_path)

        all_questions = []
        all_cands = []
        all_answers = []
        all_scores = []
        all_responses = []
        all_knowledge = []
        ids = []

        for idx, row in tqdm(df.iterrows()):
            res, res_questions, res_cands, res_answers, res_scores =\
                self.get_response_score(row['response'], row['knowledge'], gen_method, single, remove_personal)

            all_questions.extend(res_questions)
            all_cands.extend(res_cands)
            all_answers.extend(res_answers)
            all_scores.extend(res_scores)
            all_responses.extend([row['response']] * len(res_questions))
            all_knowledge.extend([row['knowledge']] * len(res_questions))
            ids.extend([idx] * len(res_questions))

            if res == self.INVALID_QUESTION:
                all_questions.extend([self.NO_VALID_QUESTIONS])
                all_cands.extend([self.NO_VALID_QUESTIONS])
                all_answers.extend([self.NO_VALID_QUESTIONS])
                all_scores.extend([self.INVALID_QUESTION])
                all_responses.extend([row['response'].lower()])
                all_knowledge.extend([row['knowledge']])
                ids.extend([idx])

            q_scores.append(res)

        if out_path != '':
            df['Q2'] = q_scores
            df = df[df.Q2 >= 0]
            df.to_csv(out_path)

        if save_steps:
            data = {'id': ids, 'response': all_responses, 'cand': all_cands, 'question': all_questions, 'knowledge': all_knowledge,
                    'knowledge_ans': all_answers, 'score': all_scores}
            steps_df = pd.DataFrame(data=data)
            steps_df.to_csv(out_path[:-4] + '.steps.csv')

        valid_scores = [s for s in q_scores if s != -1]
        print("total with at least 1 valid question:", len(valid_scores))
        print("score:", np.mean(valid_scores))

        return valid_scores


class NLI_Scorer():
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
                                predictor_name="textual_entailment", cuda_device=0)
        self.NO_ANS = '[CLS]'
        self.NO_NLI = 'NO_NLI'
        self.NO_Q = -1
        self.ENTAILMENT_SCORE = 1
        self.CONTRADICTION_SCORE = 0
        self.NEUTRAL_SCORE = 0.5

        self.in_path = args.outfile[:-4]+".steps.csv"
        self.out_path = args.outfile[:-4]+"_scores.csv"

    def get_e2e_nli_score(self, response, knowledge):

        res = self.predictor.predict(
            premise=knowledge,
            hypothesis=response
        )

        nli_label = res['label']

        if nli_label == 'entailment':  # If entails, the score is 1
            return self.ENTAILMENT_SCORE
        elif nli_label == 'contradiction':  # If contradicts, the score is 0
            return self.CONTRADICTION_SCORE
        else:
            return self.NEUTRAL_SCORE


    def get_nli_label(self, question, cand, evidence_ans):
        premise = question + ' ' + evidence_ans + '.'
        hypothesis = question + ' ' + cand + '.'

        res = self.predictor.predict(
            premise=premise,
            hypothesis=hypothesis
        )

        return res['label']


    def scores_with_nli(self):
        nli_scores = []
        f1_scores = []

        df = pd.read_csv(self.in_path)

        for _, row in df.iterrows():
            f1_score = row['score']

            evidence_answer = str(row['knowledge_ans'])

            nli_score = f1_score

            # Use NLI to determine answer similarity.
            # This is only applicable for responses that had at least one valid question generated

            if 0 <= f1_score < 1 and self.NO_ANS not in evidence_answer and evidence_answer != '' and evidence_answer != 'nan':
                f1_scores.append(f1_score)
                # If the score is 1, there is a full overlap between the
                # candidate and the predicted answer, so the score is 1
                # If there is no answer - can't run NLI, keep the original score (0)

                nli_label = self.get_nli_label(str(row['question']), str(row['cand']), evidence_answer)

                if nli_label == 'entailment':  # If entails, the score is 1
                    nli_score = self.ENTAILMENT_SCORE
                elif nli_label == 'contradiction':  # If contradicts, the score is 0
                    nli_score = self.CONTRADICTION_SCORE

            # Add fallback NLI to responses that are not covered by Q2 (no questions generated)
            elif f1_score == self.NO_Q:
                nli_fallback = self.get_e2e_nli_score(str(row['response']), str(row['knowledge']).lower())
                nli_score = nli_fallback
                f1_scores.append(nli_fallback)
            else:
                f1_scores.append(f1_score)

            nli_scores.append(nli_score)

        df['q2_score'] = nli_scores
        df['q2_no_nli'] = f1_scores
        return df


    def aggregate_per_response(self, df, for_systems_simulation=False):
        f1_scores_by_id = dict()
        nli_scores_by_id = dict()
        knowledge_by_id = dict()
        response_by_id = dict()
        label_by_id = dict()

        for _, row in df.iterrows():
            idx = row['id']
            f1_score = row['q2_no_nli']
            nli_score = row['q2_score']

            if idx in f1_scores_by_id:
                f1_scores_by_id[idx].append(f1_score)
                nli_scores_by_id[idx].append(nli_score)
            else:
                f1_scores_by_id[idx] = [f1_score]
                nli_scores_by_id[idx] = [nli_score]
                response_by_id[idx] = row['response']
                knowledge_by_id[idx] = row['knowledge']
                if for_systems_simulation:
                    label_by_id[idx] = row['label']

        mean_f1_scores = []
        mean_nli_scores = []
        responses = []
        knowledge = []
        labels = []

        for idx in f1_scores_by_id.keys():
            mean_f1_scores.append(np.mean(f1_scores_by_id[idx]))
            mean_nli_scores.append(np.mean(nli_scores_by_id[idx]))
            responses.append(response_by_id[idx])
            knowledge.append(knowledge_by_id[idx])
            if for_systems_simulation:
                labels.append(label_by_id[idx])

        print('Q2:', np.mean(mean_nli_scores))
        print('Q2, no nli:', np.mean(mean_f1_scores))
        data = {'id': list(f1_scores_by_id.keys()), 'response': responses, 'knowledge': knowledge,
                'Q2_no_nli': mean_f1_scores, 'Q2': mean_nli_scores}

        res_df = pd.DataFrame(data=data)
        if for_systems_simulation:
            res_df['label'] = labels

        res_df.to_csv(self.out_path)

        #Added
        log_out_path = self.out_path[:-4] + ".json"
        res_json = {"Q2":np.mean(mean_nli_scores), "Q2 (no nli)":np.mean(mean_f1_scores)}
        with open(log_out_path, "w", encoding="utf-8") as f:
            json.dump(res_json, f, indent=4)


    def add_baseline_e2e_nli(self):
        df = pd.read_csv(self.in_path)
        e2e_nli_scores = []
        for _, row in df.iterrows():
            e2e_nli_scores.append(self.get_e2e_nli_score(str(row['response']), str(row['knowledge']).lower()))
        df['e2e_nli'] = e2e_nli_scores
        df.to_csv(self.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #For Q2 
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to a csv file containing dialogue model outputs.")
    parser.add_argument("--gen_method", type=str, required=False, choices=['greedy', 'beam', 'sampling'], default="beam",
                        help="Decoding method for question generation.")
    parser.add_argument("--q_per_cand", type=str, choices=['single', 'multi'], default="single", required=False,
                        help="Take only one question per candidate when using beam/sampling for decoding")
    parser.add_argument("--personal", type=str, choices=['keep', 'remove'], default='remove', required=False,
                        help="Whether to remove personal questions.")
    parser.add_argument("--outfile", type=str, default='', required=False, help="Path to an output file")
    parser.add_argument("--save_steps", default=False, action="store_true", help="Whether to save all pipeline steps")
    #For NLI
    # parser.add_argument("--infile_NLI", type=str, default='', required=False,
    #                     help="Path to a csv file containing token-level f1 scores.")
    # parser.add_argument("--outfile_NLI", type=str, default='', required=False, help="Path to an output file")
    parser.add_argument("--task", type=str, required=False, choices=['span_comparison', 'e2e'], default="span_comparison",
                        help="The desired task.")
    parser.add_argument("--for_systems_simulation", default=False, action="store_true", help="Whether the input was the cross annotation data, used for the systems simulation "
                             "experiments.")
    args = parser.parse_args()

    if args.q_per_cand == 'single':
        single_q = True
    else:
        single_q = False

    if args.personal == 'remove':
        rm_personal = True
    else:
        rm_personal = False

    p = pathlib.Path(args.outfile)
    p.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading QA and QG models to compute the Q2 score...")
    Q2 = Q2_Scorer()
    logging.info("Computing the Q2 scores...")
    Q2.calc_scores(args.infile, args.gen_method, single=single_q, remove_personal=rm_personal,
                out_path=args.outfile, save_steps=args.save_steps)
    logging.info("Loading hte NLI model...")
    nli_scorer = NLI_Scorer(args)
    logging.info("Computing the final Q2 score...")
    if args.task == 'span_comparison':
        with_nli_df = nli_scorer.scores_with_nli()
        nli_scorer.aggregate_per_response(with_nli_df, args.for_systems_simulation)
    elif args.task == 'e2e':
        nli_scorer.add_baseline_e2e_nli()

