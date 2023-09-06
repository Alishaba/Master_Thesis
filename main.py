from pykeen.evaluation import EnsembleRankBasedEvaluator
from pykeen.evaluation import RankBasedEvaluator
from pykeen.training import TrainingCallback, NonFiniteLossError
from pykeen.datasets import get_dataset
from pykeen.pipeline import pipeline
from pykeen import predict
from pykeen.models import RotatE, TransE, DistMult, ComplEx
from pykeen.sampling import ExtendedBasicNegativeSampler, BasicNegativeSampler
from typing import Any, List
from math import cos, pi
import pandas as pd
import numpy as np
import itertools
import random
import torch
import time
import sys
import os
import math



class ModelSavingCallback(TrainingCallback):
    def __init__(self, batch_size: int, dataset_size: int, step: int, 
    max_lr: float, min_lr: float, num_snapshots: int, num_epochs: int, method: str, dataset_name: str,
    model_name: str):
        """
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.step = self._is_divisible_step(num_epochs, step)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_snapshots = num_snapshots
        self.method = method
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_epochs_passed = num_epochs - num_snapshots * step
        self.T = step * round(dataset_size/batch_size + 0.5)
        self.c = 0
    def _is_divisible_step(self, num_epochs, step):
        if not num_epochs % step == 0:
            raise ValueError("Number of epochs should be divisible \
            by number of steps")
        return step
    def calc_lr(self):
        # lr = self.min_lr + 0.5*(self.max_lr - self.min_lr)* \
        # (1+cos(pi*self.c/self.T)) # For MMCCLR
        lr = (self.max_lr/2)*(cos((pi*((self.c)%math.ceil(self.T)))/ \
        math.ceil(self.T)) + 1) # For CCA
        return lr
    def on_batch(self, epoch: int, batch, 
                 batch_loss: float, **kwargs: Any) -> None:
        if epoch > self.num_epochs_passed:
            self.optimizer.param_groups[0]['lr'] = self.calc_lr()
    def post_batch(self, epoch: int, batch, **kwargs: Any) -> None:
        self.c += 1
    def post_epoch(self, epoch: int, epoch_loss: float,
                   **kwargs: Any) -> None:      
        print(epoch, epoch_loss, self.optimizer.param_groups[0]['lr'])
        if epoch == self.num_epochs_passed:
            self.c = 0
        if epoch % self.step == 0 and epoch > self.num_epochs_passed:
            self.c = 0
            torch.save(self.model, f'./models/trained_model_{self.dataset_name}_{self.model_name}_{self.method}_{epoch}.pkl')


class NoModelSavingCallback(TrainingCallback):
    def __init__(self, batch_size: int, dataset_size: int, step: int, 
    max_lr: float, min_lr: float, num_snapshots: int, num_epochs: int, method: str, dataset_name: str,
    model_name: str):
        """
        """
        super().__init__()
        self.method = method
        self.dataset_name = dataset_name
        self.model_name = model_name
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:
      
        print(epoch, epoch_loss, self.optimizer.param_groups[0]['lr'])

    def post_train(self, losses: List[float], **kwargs: Any) -> None:
        torch.save(self.model,f'./models/trained_model_{self.dataset_name}_{self.model_name}_{self.method}.pkl')


def tune_baseline(training, validation, test, holdout, conf, learning_rates):
    model = mos[mn]
    num_epochs, dtt, dte = [], [], []
    both, head, tail = [], [], []
    h10both, h10head, h10tail = [], [], []
    h3both, h3head, h3tail = [], [], []
    h1both, h1head, h1tail = [], [], []
    bad_lr = []
    for lr in learning_rates:
        print(f'LEARNING RATE {lr}')
        conf.update({"optimizer_kwargs": dict(lr=lr)})
        try:
            start_time = time.time()
            pipeline_result = run_pipeline(model, training, validation, test, conf, method=f'baseline_{lr}')
        except NonFiniteLossError:
            bad_lr.append(lr)
            print(f'Skipping learning rate {lr} due to NonFiniteLossError {bad_lr}')
            
            continue
        dtt.append(time.time() - start_time)
        num_epochs.append(pipeline_result.stopper.best_epoch)
        start_time = time.time()
        res, _, _, _ = run_evaluation(test, [training.mapped_triples, validation.mapped_triples], conf, method=f'baseline_{lr}')
        dte.append(time.time() - start_time)
        MRR_head=res[(res.Side=='head') & (res.Type=='optimistic') & (res.Metric==conf['metrics'])]['Value'].values[0],
        MRR_tail=res[(res.Side=='tail') & (res.Type=='optimistic') & (res.Metric==conf['metrics'])]['Value'].values[0], 
        MRR_both=res[(res.Side=='both') & (res.Type=='optimistic') & (res.Metric==conf['metrics'])]['Value'].values[0],

        HITS10_head=res[(res.Side=='head') & (res.Type=='optimistic') & (res.Metric=='hits_at_10')]['Value'].values[0],
        HITS10_tail=res[(res.Side=='tail') & (res.Type=='optimistic') & (res.Metric=='hits_at_10')]['Value'].values[0], 
        HITS10_both=res[(res.Side=='both') & (res.Type=='optimistic') & (res.Metric=='hits_at_10')]['Value'].values[0],

        HITS3_head=res[(res.Side=='head') & (res.Type=='optimistic') & (res.Metric=='hits_at_3')]['Value'].values[0],
        HITS3_tail=res[(res.Side=='tail') & (res.Type=='optimistic') & (res.Metric=='hits_at_3')]['Value'].values[0], 
        HITS3_both=res[(res.Side=='both') & (res.Type=='optimistic') & (res.Metric=='hits_at_3')]['Value'].values[0],

        HITS1_head=res[(res.Side=='head') & (res.Type=='optimistic') & (res.Metric=='hits_at_1')]['Value'].values[0],
        HITS1_tail=res[(res.Side=='tail') & (res.Type=='optimistic') & (res.Metric=='hits_at_1')]['Value'].values[0], 
        HITS1_both=res[(res.Side=='both') & (res.Type=='optimistic') & (res.Metric=='hits_at_1')]['Value'].values[0],

        both.append(MRR_both)
        head.append(MRR_head)
        tail.append(MRR_tail)
        h10both.append(HITS10_both)
        h10head.append(HITS10_head)
        h10tail.append(HITS10_tail)
        h3both.append(HITS3_both)
        h3head.append(HITS3_head)
        h3tail.append(HITS3_tail)
        h1both.append(HITS1_both)
        h1head.append(HITS1_head)
        h1tail.append(HITS1_tail)
        print(f'MRR for model with lr {lr}: {MRR_head}, {MRR_tail}, {MRR_both}')
        
    for blr in bad_lr:
        learning_rates.remove(blr)
    max_i = np.argmax(both)
    print(f'Best lr is {learning_rates[max_i]} and number of epochs is {num_epochs[max_i]}')
    res_test, _, _, _ = run_evaluation(holdout, [training.mapped_triples, validation.mapped_triples, test.mapped_triples], conf, method=f'baseline_{learning_rates[max_i]}')
    MRR_head_test=res_test[(res_test.Side=='head') & (res_test.Type=='optimistic') & (res_test.Metric==conf['metrics'])]['Value'].values[0],
    MRR_tail_test=res_test[(res_test.Side=='tail') & (res_test.Type=='optimistic') & (res_test.Metric==conf['metrics'])]['Value'].values[0], 
    MRR_both_test=res_test[(res_test.Side=='both') & (res_test.Type=='optimistic') & (res_test.Metric==conf['metrics'])]['Value'].values[0],

    HITS10_head_test=res_test[(res_test.Side=='head') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_10')]['Value'].values[0],
    HITS10_tail_test=res_test[(res_test.Side=='tail') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_10')]['Value'].values[0], 
    HITS10_both_test=res_test[(res_test.Side=='both') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_10')]['Value'].values[0],

    HITS3_head_test=res_test[(res_test.Side=='head') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_3')]['Value'].values[0],
    HITS3_tail_test=res_test[(res_test.Side=='tail') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_3')]['Value'].values[0], 
    HITS3_both_test=res_test[(res_test.Side=='both') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_3')]['Value'].values[0],

    HITS1_head_test=res_test[(res_test.Side=='head') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_1')]['Value'].values[0],
    HITS1_tail_test=res_test[(res_test.Side=='tail') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_1')]['Value'].values[0], 
    HITS1_both_test=res_test[(res_test.Side=='both') & (res_test.Type=='optimistic') & (res_test.Metric=='hits_at_1')]['Value'].values[0],

    best_model_results = dict(MRR_head=MRR_head_test, MRR_tail=MRR_tail_test, MRR_both=MRR_both_test,
                              HITS10_head=HITS10_head_test, HITS10_tail=HITS10_tail_test, HITS10_both=HITS10_both_test, 
                              HITS3_head=HITS3_head_test, HITS3_tail=HITS3_tail_test, HITS3_both=HITS3_both_test, 
                              HITS1_head=HITS1_head_test, HITS1_tail=HITS1_tail_test, HITS1_both=HITS1_both_test)
    return learning_rates[max_i], num_epochs[max_i], best_model_results, dtt[max_i], dte[max_i]


def run_pipeline(model, training, validation, test, conf, method, relative_delta=0.000001):
    callbacks= ModelSavingCallback if method in ['paper', 'proposed'] else NoModelSavingCallback
    negative_sampler = ExtendedBasicNegativeSampler if method in ['paper', 'proposed'] and conf['sampler'] == 'extended' else BasicNegativeSampler
    negative_sampler_kwargs = None if method not in ['paper', 'proposed'] or conf['sampler'] == 'basic' else dict(
        num_batches=math.floor(training.num_triples/conf['batch_size']),
        models_to_load=conf['models_to_load'],
        num_epochs=conf['num_epochs'],
        step=conf['step'],
        dataset_name=conf['dataset_name'],
        model_name=conf['model_name'],
        method=method,
        )
    stopper = None if method in ['paper', 'proposed'] else 'early'
    pipeline_result = pipeline(
    training=training,
    validation=validation,
    testing=test,
    random_seed=10,
    model=model,
    model_kwargs=dict(embedding_dim=conf['embedding_dim']),
    training_loop='sLCWA',
    optimizer=conf['optimizer'],
    optimizer_kwargs=conf['optimizer_kwargs'],
    training_kwargs=dict(
        batch_size=conf['batch_size'],
        drop_last=True,
        num_epochs=conf['num_epochs'],
        callbacks=callbacks,
        callback_kwargs=dict(
            batch_size=conf['batch_size'],
            dataset_size=training.num_triples,
            max_lr=conf['max_lr'],
            min_lr=conf['min_lr'],
            step=conf['step'],
            num_snapshots=conf['num_snapshots'],
            num_epochs=conf['num_epochs'],
            method=method,
            dataset_name=conf['dataset_name'],
            model_name=conf['model_name']
        ),
    ),
    stopper=stopper,
    stopper_kwargs=dict(relative_delta=relative_delta, evaluation_batch_size=conf['batch_size'], metric=conf['metrics']),
    negative_sampler=negative_sampler,
    negative_sampler_kwargs=negative_sampler_kwargs
    )
    print('Saving')
    torch.save(pipeline_result.model,f"./models/trained_model_{conf['dataset_name']}_{conf['model_name']}_{method}.pkl")
    print('Saving done!')
    return pipeline_result


def run_evaluation(test_set, additional_filter_triples, conf, method, losses=None):
    if method in ['paper', 'proposed']:
        last_m = conf['models_to_load']
        start_time = time.time()
        model = [torch.load(f"./models/trained_model_{conf['dataset_name']}_{conf['model_name']}_{method}_{conf['num_epochs'] - i * conf['step']}.pkl") for i in range(0, last_m)]
        loading_time = time.time() - start_time
        s = [f"./models/trained_model_{conf['dataset_name']}_{conf['model_name']}_{method}_{conf['num_epochs'] - i * conf['step']}.pkl" for i in range(0, last_m)]
        print(f' loading{s}')
        evaluator = EnsembleRankBasedEvaluator(metrics=conf['metrics'])
        both, head, tail = [], [], []
        h10both, h10head, h10tail = [], [], []
        h3both, h3head, h3tail = [], [], []
        h1both, h1head, h1tail = [], [], []
        for i, m in enumerate(model):
            m_res = evaluator.evaluate(m, test_set.mapped_triples, batch_size=conf['batch_size'], 
                                         additional_filter_triples=additional_filter_triples).to_df()
            head.append(m_res[(m_res.Side=='head') & (m_res.Type=='optimistic') & (m_res.Metric==c['metrics'])]['Value'].values[0])
            tail.append(m_res[(m_res.Side=='tail') & (m_res.Type=='optimistic') & (m_res.Metric==c['metrics'])]['Value'].values[0]) 
            both.append(m_res[(m_res.Side=='both') & (m_res.Type=='optimistic') & (m_res.Metric==c['metrics'])]['Value'].values[0])
            h10head.append(m_res[(m_res.Side=='head') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_10')]['Value'].values[0])
            h10tail.append(m_res[(m_res.Side=='tail') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_10')]['Value'].values[0])
            h10both.append(m_res[(m_res.Side=='both') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_10')]['Value'].values[0])
            h3head.append(m_res[(m_res.Side=='head') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_3')]['Value'].values[0])
            h3tail.append(m_res[(m_res.Side=='tail') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_3')]['Value'].values[0])
            h3both.append(m_res[(m_res.Side=='both') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_3')]['Value'].values[0])
            h1head.append(m_res[(m_res.Side=='head') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_1')]['Value'].values[0])
            h1tail.append(m_res[(m_res.Side=='tail') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_1')]['Value'].values[0])
            h1both.append(m_res[(m_res.Side=='both') & (m_res.Type=='optimistic') & (m_res.Metric=='hits_at_1')]['Value'].values[0])
            print(f'MRR for model {s[i]}: {head[i]}, {tail[i]}, {both[i]}')

        max_i = np.argmax(both)
        print(f'Best model is {s[max_i]}')
        best_model_results = dict(MRR_head=head[max_i], MRR_tail=tail[max_i], MRR_both=both[max_i],
                              HITS10_head=h10head[max_i], HITS10_tail=h10tail[max_i], HITS10_both=h10both[max_i], 
                              HITS3_head=h3head[max_i], HITS3_tail=h3tail[max_i], HITS3_both=h3both[max_i], 
                              HITS1_head=h1head[max_i], HITS1_tail=h1tail[max_i], HITS1_both=h1both[max_i])

        results = {}
        eval_times = {}
        for w in ['loss', 'MRR', 'equal', 'descending', 'borda']:
            for normalize in ['MinMax', 'Standard', None]:
                start_time = time.time()
                weights, borda = get_weights(conf, models_to_load, losses, both, w)

                print(weights)
                print(f"borda {borda}")
                
                results[f'{w}_{normalize}'] = evaluator.evaluate(model, test_set.mapped_triples, batch_size=conf['batch_size'], 
                                        additional_filter_triples=additional_filter_triples, weights=weights, borda=borda, normalize=normalize).to_df()
                evaluation_time = time.time() - start_time
                eval_times[f'{w}_{normalize}'] = evaluation_time + loading_time
    else:
        model = torch.load(f"./models/trained_model_{conf['dataset_name']}_{conf['model_name']}_{method}.pkl") 
        
        evaluator = RankBasedEvaluator(metrics=conf['metrics'])
        
        results = evaluator.evaluate(model, test_set.mapped_triples, batch_size=conf['batch_size'],
                                     additional_filter_triples=additional_filter_triples).to_df()
        best_model_results = dict(MRR_head='', MRR_tail='', MRR_both='', HITS10_head='', HITS10_tail='', HITS10_both='', 
                                  HITS3_head='', HITS3_tail='', HITS3_both='', HITS1_head='', HITS1_tail='', HITS1_both='')
        eval_times = {}
    return results, model, best_model_results, eval_times

def get_weights(conf, models_to_load, losses, both, w):
    borda = False
    if w=='loss':
        indices = [conf['num_epochs'] - i * conf['step'] for i in range(0, models_to_load)]
        _losses = [losses[int(i) - 1] for i in indices]
        weights = [max(_losses) - i + min(_losses) for i in _losses]
        print(pipeline_result.losses)
        print(indices)
    elif w=='MRR':
        weights = both
    elif w=='equal':
        weights = None
    elif w=='descending':
        weights = [models_to_load - i for i in range(0,models_to_load)]
    elif w=='borda':
        weights = None
        borda = True
    return weights, borda


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return pd.DataFrame(list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values())))


def update_df(df, c, best_model_results, training_time, evaluation_time, method, weights, res_MRR, res_HITS10, res_HITS3, res_HITS1, avg_corr, min_corr, max_corr):
    df = df.append(dict(
                    dataset=c['dataset_name'], 
                    model=c['model_name'], 
                    method=method, 
                    sampler=c['sampler'],
                    batch_size=c['batch_size'], 
                    weights=weights,
                    step=c['step'], 
                    epochs=c['num_epochs'], 
                    num_periods=c['num_periods'],
                    num_snapshots=c['num_snapshots'], 
                    models_to_load=c['models_to_load'],
                    optimizer=c['optimizer'], 
                    max_lr=c['max_lr'], 
                    min_lr=c['min_lr'], 
                    training_time=training_time,
                    evaluation_time=evaluation_time,
                    embedding_dim=c['embedding_dim'],
                    HITS10_head=res_HITS10[(res_HITS10.Side=='head') & (res_HITS10.Type=='optimistic')]['Value'].values[0],
                    HITS10_tail=res_HITS10[(res_HITS10.Side=='tail') & (res_HITS10.Type=='optimistic')]['Value'].values[0],
                    HITS10_both=res_HITS10[(res_HITS10.Side=='both') & (res_HITS10.Type=='optimistic')]['Value'].values[0],
                    HITS3_head=res_HITS3[(res_HITS3.Side=='head') & (res_HITS3.Type=='optimistic')]['Value'].values[0],
                    HITS3_tail=res_HITS3[(res_HITS3.Side=='tail') & (res_HITS3.Type=='optimistic')]['Value'].values[0],
                    HITS3_both=res_HITS3[(res_HITS3.Side=='both') & (res_HITS3.Type=='optimistic')]['Value'].values[0],
                    HITS1_head=res_HITS1[(res_HITS1.Side=='head') & (res_HITS1.Type=='optimistic')]['Value'].values[0],
                    HITS1_tail=res_HITS1[(res_HITS1.Side=='tail') & (res_HITS1.Type=='optimistic')]['Value'].values[0],
                    HITS1_both=res_HITS1[(res_HITS1.Side=='both') & (res_HITS1.Type=='optimistic')]['Value'].values[0],
                    MRR_head=res_MRR[(res_MRR.Side=='head') & (res_MRR.Type=='optimistic')]['Value'].values[0],
                    MRR_tail=res_MRR[(res_MRR.Side=='tail') & (res_MRR.Type=='optimistic')]['Value'].values[0], 
                    MRR_both=res_MRR[(res_MRR.Side=='both') & (res_MRR.Type=='optimistic')]['Value'].values[0],
                    BestModel_HITS10_head=best_model_results['HITS10_head'],
                    BestModel_HITS10_tail=best_model_results['HITS10_tail'],
                    BestModel_HITS10_both=best_model_results['HITS10_both'],
                    BestModel_HITS3_head=best_model_results['HITS3_head'],
                    BestModel_HITS3_tail=best_model_results['HITS3_tail'],
                    BestModel_HITS3_both=best_model_results['HITS3_both'],
                    BestModel_HITS1_head=best_model_results['HITS1_head'],
                    BestModel_HITS1_tail=best_model_results['HITS1_tail'],
                    BestModel_HITS1_both=best_model_results['HITS1_both'],
                    BestModel_MRR_head=best_model_results['MRR_head'],
                    BestModel_MRR_tail=best_model_results['MRR_tail'],
                    BestModel_MRR_both=best_model_results['MRR_both'],
                    avg_corr=avg_corr,
                    min_corr=min_corr,
                    max_corr=max_corr
                    ), ignore_index=True)
        
    return df


def update_df_baseline(df, c, best_model_results, training_time, evaluation_time):
    df = df.append(dict(
                    dataset=c['dataset_name'], 
                    model=c['model_name'], 
                    method='baseline', 
                    sampler='',
                    batch_size=c['batch_size'], 
                    weights='',
                    step=c['step'], 
                    epochs=c['num_epochs'], 
                    num_periods=c['num_periods'],
                    num_snapshots=1,
                    models_to_load=1,
                    optimizer=c['optimizer'], 
                    max_lr=c['max_lr'], 
                    min_lr=c['min_lr'], 
                    training_time=training_time,
                    evaluation_time=evaluation_time,
                    embedding_dim=c['embedding_dim'],
                    HITS10_head=best_model_results['HITS10_head'][0],
                    HITS10_tail=best_model_results['HITS10_tail'][0],
                    HITS10_both=best_model_results['HITS10_both'][0],
                    HITS3_head=best_model_results['HITS3_head'][0],
                    HITS3_tail=best_model_results['HITS3_tail'][0],
                    HITS3_both=best_model_results['HITS3_both'][0],
                    HITS1_head=best_model_results['HITS1_head'][0],
                    HITS1_tail=best_model_results['HITS1_tail'][0],
                    HITS1_both=best_model_results['HITS1_both'][0],
                    MRR_head=best_model_results['MRR_head'][0],
                    MRR_tail=best_model_results['MRR_tail'][0], 
                    MRR_both=best_model_results['MRR_both'][0],
                    BestModel_HITS10_head='',
                    BestModel_HITS10_tail='',
                    BestModel_HITS10_both='',
                    BestModel_HITS3_head='',
                    BestModel_HITS3_tail='',
                    BestModel_HITS3_both='',
                    BestModel_HITS1_head='',
                    BestModel_HITS1_tail='',
                    BestModel_HITS1_both='',
                    BestModel_MRR_head='',
                    BestModel_MRR_tail='',
                    BestModel_MRR_both='',
                    avg_corr='',
                    min_corr='', 
                    max_corr=''
                ), ignore_index=True)
            
    return df


def get_corr_values(models, triples):
  rank_predictions_dict = {}
  for i, model in enumerate(models):
      df = predict.predict_triples(model=model, triples=triples)
      df = df.process().df
      rank_predictions_dict[i] = df['score']
  corr_matrix = pd.DataFrame(rank_predictions_dict).corr()
  corr_tri = np.triu(corr_matrix.values)
  pairwise_corr = corr_tri[np.triu_indices(len(models), k = 1)]
  avg_ = np.average(pairwise_corr)
  min_ = np.min(pairwise_corr)
  max_ = np.max(pairwise_corr)
  return avg_, min_, max_



# The script starts here
random.seed(10)
learning_rates = [10, 1, 0.1, 0.01]
batch_size = 128
mtl = dict(num_periods_5 = [2, 3, 4, 5], num_periods_10 = [2, 3, 4, 5, 6, 7, 8, 9, 10])
n_periods = [5, 10]

if os.path.exists('results.csv'):
    df = pd.read_csv('results.csv')
    print('Loading the results file')
else:
    df = pd.DataFrame(columns=['dataset', 'model', 'method', 'sampler', 'batch_size', 'weights'
    'step', 'epochs', 'num_periods', 'num_snapshots', 'models_to_load', 'optimizer', 'max_lr', 'min_lr', 'training_time', 'evaluation_time', 'embedding_dim', 
    'HITS10_head', 'HITS10_tail', 'HITS10_both', 'HITS3_head', 'HITS3_tail', 'HITS3_both', 'HITS1_head', 'HITS1_tail', 'HITS1_both', 'MRR_head', 'MRR_tail', 'MRR_both',
    'BestModel_HITS10_head', 'BestModel_HITS10_tail', 'BestModel_HITS10_both', 'BestModel_HITS3_head', 'BestModel_HITS3_tail', 'BestModel_HITS3_both', 'BestModel_HITS1_head', 'BestModel_HITS1_tail', 'BestModel_HITS1_both', 'BestModel_MRR_head', 'BestModel_MRR_tail', 'BestModel_MRR_both',
    'avg_corr', 'min_corr', 'max_corr'])
    print('Creating a new results file')

if os.path.exists('configurations.csv'):
    confs = pd.read_csv('configurations.csv')
    print('Loading the configurations file')
else:
    conf_lists = dict(
        dn =['WN18RR', 'FB15k237', 'DBpedia50', 'AristoV4'],
        mn = ['RotatE', 'TransE', 'DistMult', 'ComplEx'], 
        optimizer = ['SGD', 'Adagrad', 'Adam'],
        embedding_dim = [128, 64],
    )
    confs = dict_product(conf_lists)
    confs.loc[:, 'done'] = 0
    confs.loc[:, 'subdone'] = 0
    confs.to_csv('configurations.csv', index=False)
    print('Creating a new configurations file')

for i, row in confs.iterrows():
    dn, mn, optimizer, embedding_dim, done, subdone = row
    if done == 1:
        print('Skipping this config ', dn, mn, optimizer, embedding_dim, done, subdone)
        continue
    elif subdone < 781 and subdone > 0:
        print('Continuing to process this config ', dn, mn, optimizer, embedding_dim, done, subdone)
        learning_rate = df[(df.dataset==dn) & (df.model==mn) & (df.optimizer==optimizer) & (df.embedding_dim==embedding_dim) & (df.method=='baseline')]['max_lr'].values[0]
        new_num_epochs = df[(df.dataset==dn) & (df.model==mn) & (df.optimizer==optimizer) & (df.embedding_dim==embedding_dim) & (df.method=='baseline')]['epochs'].values[0]
        print(f'learning_rate {learning_rate} and new_num_epochs {new_num_epochs}')
    print('Starting to process this config ', dn, mn, optimizer, embedding_dim, done, subdone)

    c = dict(
            dataset_name = dn,
            model_name = mn,
            embedding_dim=embedding_dim,
            batch_size = batch_size,
            num_epochs = 1000,
            num_periods=None,
            max_lr = None,
            min_lr = None,
            step = None,
            sampler=None,
            num_snapshots = None,
            models_to_load=None,
            optimizer=optimizer,
            metrics='inverse_harmonic_mean_rank',
            weights=None
    )

    dataset = get_dataset(dataset=dn)
    training = dataset.training
    validation = dataset.validation
    test = dataset.testing
    training, holdout = training.split(ratios=(0.95, 0.05), random_state=10)
    mos = dict(RotatE=RotatE(triples_factory=training, random_seed=10, embedding_dim=64), TransE=TransE(triples_factory=training, random_seed=10, embedding_dim=64), 
                             DistMult=DistMult(triples_factory=training, random_seed=10, embedding_dim=64), ComplEx=ComplEx(triples_factory=training, random_seed=10, embedding_dim=64))
    print(f'training {len(training.mapped_triples)}')
    print(f'validation {len(validation.mapped_triples)}')
    print(f'test {len(test.mapped_triples)}')
    print(f'holdout {len(holdout.mapped_triples)}')
    print(f'training entities {training.num_entities}')

    counter = 0
    if counter >= subdone:
        learning_rate, new_num_epochs, best_model_results, training_time, evaluation_time = tune_baseline(training, validation, test, holdout, c, learning_rates)
        c.update({"num_epochs": new_num_epochs})
        c.update({"optimizer_kwargs": dict(lr=learning_rate)})
        c.update({"max_lr": learning_rate})
        c.update({"min_lr": learning_rate})
        df = update_df_baseline(df, c, best_model_results, training_time, evaluation_time)
        df.to_csv('results.csv', index=False)
        confs.loc[i, 'subdone'] = 1
        confs.to_csv('configurations.csv', index=False)
    else:
        print(f'skipping exp 1')

    new_num_epochs = max(new_num_epochs,20)
    counter += 1
    c.update({"num_epochs": new_num_epochs})
    c.update({"optimizer_kwargs": dict(lr=learning_rate)})
    for method in ['paper', 'proposed']:
        for num_periods in n_periods:
            for sampler in ['basic', 'extended']:
                print(sampler)
                c.update({"num_periods": num_periods})
                c.update({"step": int(new_num_epochs/num_periods)})
                c.update({"sampler": sampler})
                
                c.update({"max_lr": learning_rate})
                c.update({"min_lr": learning_rate/100})
                
                for models_to_load in mtl[f'num_periods_{num_periods}']:
                    for snapshots_embedding_dim in [int(embedding_dim/models_to_load)]:
                        mos = dict(RotatE=RotatE(triples_factory=training, random_seed=10, embedding_dim=snapshots_embedding_dim), TransE=TransE(triples_factory=training, random_seed=10, embedding_dim=snapshots_embedding_dim), 
                                    DistMult=DistMult(triples_factory=training, random_seed=10, embedding_dim=snapshots_embedding_dim), ComplEx=ComplEx(triples_factory=training, random_seed=10, embedding_dim=snapshots_embedding_dim))
                        if counter < subdone:
                            counter += 15
                            print(f'skipping exp {counter}')
                            continue
                        print(f'starting from exp {counter}')
                        c.update({"embedding_dim": snapshots_embedding_dim})
                        num_snapshots = num_periods if method == 'paper' else models_to_load
                        c.update({"num_snapshots": num_snapshots})
                        c.update({"models_to_load": models_to_load})
                        model = mos[mn]
                        start_time = time.time()
                        pipeline_result = run_pipeline(model, training, validation, test, c, method=method)
                        training_time = time.time() - start_time
                        results, models, best_model_results, eval_times = run_evaluation(holdout, [training.mapped_triples, validation.mapped_triples, test.mapped_triples], c, method=method, losses=pipeline_result.losses)
                        avg_corr, min_corr, max_corr = get_corr_values(models, training.mapped_triples)
                        del models
                        for w in results:
                            res = results[w]
                            res_MRR = res[res.Metric==c['metrics']]
                            res_HITS10 = res[res.Metric=='hits_at_10']
                            res_HITS3 = res[res.Metric=='hits_at_3']
                            res_HITS1 = res[res.Metric=='hits_at_1']
                            evaluation_time = eval_times[w]
                            df = update_df(df, c, best_model_results, training_time, evaluation_time, method, w, res_MRR, res_HITS10, res_HITS3, res_HITS1, avg_corr, min_corr, max_corr )
                        counter += len(results)
                        confs.loc[i, 'subdone'] = counter
                        confs.to_csv('configurations.csv', index=False)
                        df.to_csv('results.csv', index=False)
                        
                        dir_ = './models'
                        for i_file in os.listdir(dir_):
                            if 'trained_model' in i_file:
                                print(f'Deleting {i_file}')
                                os.remove(os.path.join(dir_, i_file))

    confs.loc[i, 'done'] = 1
    confs.to_csv('configurations.csv', index=False)

