'''

Generic patent representation evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from patenteval import utils
from patenteval.patent import PriorArtEval, IPC_ClassificationEval, IPC_KNNEval, SingularSpectrumEval, UniformityEval, AlignmentEval, TopologyEval

class PE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold
        params.max_input_len = 512 if 'max_input_len' not in params else params.max_input_len

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['IPC-Classification', 'IPC-KNN', 'PriorArt', 'SingularSpectrum', 'Uniformity', 'Alignment', 'Topology']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        if self.params.final_eval:
            prior_art_path = self.params.task_path + '/downstream/perf200'
        else:
            prior_art_path = self.params.task_path + '/downstream/perf20'

        if name == 'SingularSpectrum':
            self.evaluation = SingularSpectrumEval(prior_art_path, params=self.params)
        elif name == 'Uniformity':
            self.evaluation = UniformityEval(prior_art_path, params=self.params)
        elif name == 'Alignment':
            self.evaluation = AlignmentEval(prior_art_path, params=self.params)
        elif name == 'PriorArt':
            self.evaluation = PriorArtEval(prior_art_path, params=self.params)
        elif name == 'IPC-KNN':
            self.evaluation = IPC_KNNEval(tpath + '/downstream/IPC-Classification', params=self.params)
        elif name == 'IPC-Classification':
            self.evaluation = IPC_ClassificationEval(tpath + '/downstream/IPC-Classification', params=self.params)
        elif name == 'Topology':
            self.evaluation = TopologyEval(prior_art_path, params=self.params)
        else:
            raise ValueError('Evaluation name not recognized: %s' % name)


        self.params.current_task = name
        # self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
