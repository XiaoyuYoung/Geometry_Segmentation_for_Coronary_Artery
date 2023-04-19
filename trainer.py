from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pprint import pprint, pformat
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import traceback
import logging
import torch
import time
import copy
import os


def source_import(file_path):
    import importlib
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_dice(y_pred, y_true):
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    intersection = np.count_nonzero(y_pred & y_true)

    size_i1 = np.count_nonzero(y_pred)
    size_i2 = np.count_nonzero(y_true)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def get_chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    from sklearn.neighbors import NearestNeighbors

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


class Trainer(object):
    def __init__(self, config):
        self.config = config

    def init_loggers(self):
        LOG_FORMAT = "%(asctime)s  [%(levelname)s]:  %(message)s"
        DAT_FORMAT = "%Y-%m-%d %H:%M:%S"

        logfile = "{}.log".format(self.tag)

        F = open(logfile, encoding="utf-8", mode="a")
        logging.basicConfig(stream=F, level=logging.INFO, format=LOG_FORMAT, datefmt=DAT_FORMAT)

    def init_tfboard(self):
        self.tboard_writer = SummaryWriter()

    def load_dataset(self):

        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '----------------  Data Loading ----------------\n'
            '-----------------------------------------------\n'
        )

        params = self.config['dataset']

        dataset_param = list(params['defin_parm'].values())
        train_dataset, valid_dataset = source_import(params['defin_path']).get_dataset(*dataset_param)

        train_nums = train_dataset.__len__()
        valid_nums = valid_dataset.__len__()

        batch_size = params['batch_size']
        num_worker = params['num_worker']
        isdroplast = params['isdroplast']
        is_shuffle = params['is_shuffle']

        defin_sampler = params['defin_sampler']
        param_sampler = list(params['param_sampler'].values())

        self.show_outputs(
            'Batch  size:{}\n'
            'Worker num:{}\n'
            'Train  num:{}\n'
            "Valid  num:{}\n"
            "Drop last :{}\n"
            "Shuffle: {}\n"
                .format(batch_size, num_worker, train_nums, valid_nums, isdroplast, is_shuffle)
        )

        if defin_sampler is not None:
            sampler = source_import(defin_sampler).get_sampler(*param_sampler)(train_dataset)
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker,
                                               sampler=sampler, drop_last=isdroplast)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker,
                                               shuffle=is_shuffle, drop_last=isdroplast)

        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False)

    def init_parames(self, method, module):

        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.show_outputs("\nModule: {} loaded ! Init Method:{}".format(module._get_name(), method))

    def load_parames(self, checkpoint, module, key):
        module.load_state_dict(torch.load(checkpoint)['networker'][key])
        self.show_outputs("\nModule: {} Checkpoint: {} loaded !".format(module._get_name(), checkpoint))

    def init_network(self):
        self.show_outputs(
            '\n-----------------------------------------------\n'
            '----------------  Model Loading ---------------\n'
            '-----------------------------------------------\n'
        )

        params = self.config['network']
        self.use_cuda = (torch.cuda.is_available() and params['use_cuda'])
        self.use_parallel = (self.use_cuda and torch.cuda.device_count() > 1 and params['use_parallel'])

        def load_module(module, param):
            curr_params = param['cur_params'] if 'cur_params' in param else None
            init_method = param['int_method'] if 'int_method' in param else None
            if curr_params is not None:
                self.load_parames(checkpoint=curr_params, module=module, key=key)
            elif init_method is not None:
                self.init_parames(method=init_method, module=module)
            else:
                assert 'No initialization method for module !'

            if self.use_parallel:
                module = nn.DataParallel(module)

            if self.use_cuda:
                module = module.cuda()
            return module

        self.model = dict()
        modules_param = params['modules']
        for key, item in modules_param.items():
            module = source_import(item['defin_path']).create_model(*list(item['defin_parm'].values()))
            self.model[key] = load_module(module, param=item)

        self.criterions = dict()
        if 'criterions' in params:
            criterions_param = params['criterions']
            for key, item in criterions_param.items():
                criterion = source_import(item['defin_path']).create_loss(*list(item['defin_parm'].values()))
                self.criterions[key] = [load_module(criterion, param=item), item['weight']]

        self.show_outputs(
            'Model Structure:\n{}\n'
            'Criterion:\n{}\n'
            'Use CUDA:      {}\n'
            "Use Parallel:  {}\n"
                .format([s for s in self.model.values()], [s for s in self.criterions.values()], self.use_cuda,
                        self.use_parallel)
        )

    def init_optimer(self):
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '--------------  Optimizer Loading -------------\n'
            '-----------------------------------------------\n'
        )

        assert hasattr(self, 'model')
        params = self.config['network']

        self.optimizer = dict()
        for key, module in self.model.items():
            if 'optimizers' in params['modules'][key]:
                optim_param = params['modules'][key]['optimizers']
                if optim_param['type'] == 'Adam':
                    self.optimizer[key] = torch.optim.Adam(module.parameters(), lr=optim_param['lr'])
                elif optim_param['type'] == 'SGD':
                    momentum = optim_param['momentum']
                    weight_decay = optim_param['weight_decay']
                    self.optimizer[key] = torch.optim.SGD(
                        module.parameters(), lr=optim_param['lr'], momentum=momentum, weight_decay=weight_decay)
                else:
                    assert 'No recognized optimizer!'

                if 'cur_params' in optim_param and optim_param['cur_params'] is not None:
                    self.optimizer[key].load_state_dict(torch.load(optim_param['cur_params'])['optimizer'][key])

                    if 'lr' in optim_param:
                        self.optimizer[key].param_groups[0]['lr'] = optim_param['lr']

                    self.show_outputs("Optimizer: {} Checkpoint: {} loaded !\n".format(key, optim_param['cur_params']))

        for key, optim in self.optimizer.items():
            self.show_outputs(
                'Module: {} \nOptimizer:\n{}\n'
                    .format(key, optim)
            )

    def init_scheder(self):
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '--------------  Scheduler Loading -------------\n'
            '-----------------------------------------------\n'
        )

        assert hasattr(self, 'optimizer')
        params = self.config['network']

        self.scheduler = dict()
        for key, optim in self.optimizer.items():
            if 'schedulers' in params['modules'][key]:
                sched_param = params['modules'][key]['schedulers']
                if sched_param['type'] == 'CosineAnnealingLR':
                    half_cycle = sched_param['half_cycle']
                    eta_min = sched_param['eta_min']
                    self.scheduler[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, T_max=half_cycle, eta_min=eta_min)
                else:
                    assert 'No recognized scheduler!'

                if 'cur_params' in sched_param and sched_param['cur_params'] is not None:
                    self.scheduler[key].load_state_dict(torch.load(sched_param['cur_params'])['scheduler'][key])
                    self.show_outputs("Scheduler: {} Checkpoint: {} loaded !\n".format(key, sched_param['cur_params']))

        for key, (type, sched) in self.scheduler.items():
            self.show_outputs(
                'Module: {} \nScheduler: {} \n{}\n'
                    .format(key, type, pformat(sched.state_dict()))
            )

    def init_trainer(self):
        self.use_logger = self.config['monitor']['logger']
        self.use_pprint = self.config['monitor']['stdstream']
        self.use_tboard = self.config['monitor']['tensorboardx']

        self.tag = self.config['tag']
        if self.use_logger: self.init_loggers()
        if self.use_tboard: self.init_tfboard()
        self.show_outputs("\nconfig:\n{}\n\n".format(pformat(self.config)))

        self.load_dataset()
        self.init_network()
        self.init_optimer()
        self.init_scheder()

        self.checkpoint_mode = self.config['trainer']['checkpoint_mode']
        self.validation_step = self.config['trainer']['validation_step']
        self.cpnt_dire = self.checkpoint_mode['dire']
        self.save_mode = self.checkpoint_mode['type']

        self.thresh_gradnt = self.config['trainer']['threshold_grad']
        self.total_epoches = self.config['trainer']['total_epoches']
        self.current_epoch = self.config['trainer']['current_epoch']

        ## 保存训练器

    def save_trainer(self, path):

        os.makedirs(self.cpnt_dire, exist_ok=True)

        model_state_dict = dict()
        for key, module in self.model.items():
            if self.use_parallel: module = module.module
            model_state_dict[key] = copy.deepcopy(module.state_dict())

        optim_state_dict = dict()
        for key, optim in self.optimizer.items():
            optim_state_dict[key] = copy.deepcopy(optim.state_dict())

        sched_state_dict = dict()
        for key, (type, sched) in self.scheduler.items():
            sched_state_dict[key] = copy.deepcopy(sched.state_dict())

        torch.save(
            {"networker": model_state_dict,
             "optimizer": optim_state_dict,
             "scheduler": sched_state_dict,
             "cur_epoch": self.current_epoch,
             "best_epch": self.best_epoch,
             "best_indx": self.best_index
             },
            path
        )
        self.show_outputs("\ncheckpoint path: {}\nsave checkpoint done!\n".format(path))

    def forward(self, index, data):
        images, labels, verts = data
        images = images.float()
        labels = labels.float()
        verts = verts.float()

        if self.use_cuda:
            images = images.cuda()
            labels = labels.cuda()
            verts = verts.cuda()

        recon_img, img_feats = self.model['Unet'](images)
        pre_verts, pre_faces = self.model['Gseg'](img_feats)

        return [recon_img, pre_verts, pre_faces], [labels, verts]

    def get_loss(self, results):
        loss = 0
        predict, labels = results
        for key, (criterion, weight) in self.criterions.items():
            loss += criterion(predict, labels) * weight
        return loss

    def backward(self, loss):
        for optimizer in self.optimizer.values():
            optimizer.zero_grad()

        loss = loss.mean()
        loss.backward()

        if self.thresh_gradnt is not None:
            for module in self.model.values():
                nn.utils.clip_grad_norm_(module.parameters(), self.thresh_gradnt)

        for key, optimizer in self.optimizer.items():
            optimizer.step()
        return loss.item()

    def batrain(self, index, data):

        results = self.forward(index=index, data=data)
        loss = self.get_loss(results=results)

        pred, true = results
        pred_image, pred_verts = pred[0], pred[1][-1]
        true_image, true_verts = true[0], true[1]

        pred_image = pred_image.detach().cpu().numpy()
        pred_verts = pred_verts.detach().cpu().numpy()

        true_image = true_image.detach().cpu().numpy()
        true_verts = true_verts.detach().cpu().numpy()

        return self.backward(loss), [pred_image, pred_verts], [true_image, true_verts]

    def trainer(self):
        self.show_outputs(
            '\n\n'
            '-----------------------------------------------\n'
            '------------------  Training ------------------\n'
            '-----------------------------------------------\n'
            'Training {} Starting epoch {}/{}.'.format(self.tag, self.current_epoch, self.total_epoches)
        )
        time.sleep(0.5)
        for module in self.model.values():
            module.train()

        train_loss = 0.
        torch.cuda.empty_cache()
        y_trained_true = []
        y_trained_pred = []
        for index, data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            loss, pred, true = self.batrain(index, data)
            train_loss += loss
            y_trained_pred.append(pred)
            y_trained_true.append(true)

        self.metrics(y_pred=y_trained_pred, y_true=y_trained_true, isTrain=True)

        if self.use_tboard:
            self.tboard_writer.add_scalar('loss', train_loss, global_step=self.current_epoch)

        self.show_outputs('\n{} Epoch{} finished ! Loss: {}\n'.format(self.tag, self.current_epoch, train_loss))

    def predict(self, data):
        images = data
        if self.use_cuda:
            images = images.cuda()

        with torch.no_grad():
            reconst, img_feat = self.model['Unet'](images)
            p_verts, p_faces = self.model['Gseg'](img_feat)

        return [reconst, p_verts, p_faces]

    def valider(self):
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '-----------------  Evaluation  ----------------\n'
            '-----------------------------------------------'
        )
        time.sleep(0.5)

        for module in self.model.values():
            module.eval()

        y_true = []
        y_pred = []
        for index, data in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader)):
            images, labels, verts = data
            images = images.float()
            labels = labels.float().numpy()
            verts = verts.float().numpy()

            pred = self.predict(images)

            pred_image, pred_verts = pred[0], pred[1][-1]
            true_image, true_verts = labels, verts

            pred_image = pred_image.detach().cpu().numpy()
            pred_verts = pred_verts.detach().cpu().numpy()

            y_pred.append([pred_image, pred_verts])
            y_true.append([true_image, true_verts])

        dice, chamfer = self.metrics(y_pred=y_pred, y_true=y_true)

        if self.config['monitor']['index'] == 'dice':
            self.monitor.append(dice)
            index = dice
        elif self.config['monitor']['index'] == 'chamfer':
            index = chamfer
            self.monitor.append(chamfer)
        else:
            raise Exception

        if index < self.best_index or self.best_index < 0:
            self.best_index = index
            self.best_epoch = self.current_epoch

    def metrics(self, y_pred, y_true, isTrain=False):

        if isTrain:
            idx = np.random.randint(0, len(y_true), 2)
            y_true = [y_true[i] for i in idx]
            y_pred = [y_pred[i] for i in idx]

        dices = []
        chamf = []
        for pred, true in tqdm(zip(y_pred, y_true)):
            batch = pred[0].shape[0]
            for index in range(batch):
                pred_img, pred_msh = pred[0][index][0], pred[1][index]
                true_img, true_msh = true[0][index][0], true[1][index]

                dice = get_dice(np.asarray(pred_img > 0.5).astype(int), true_img)
                chaf = get_chamfer_distance(pred_msh, true_msh)

                dices.append(dice)
                chamf.append(chaf)

        self.show_outputs("\nVal num:{}\ndice:{}\nchamfer distance:{}\n".format(
            batch, np.mean(dices), np.mean(chamf)))

        if hasattr(self, 'use_tboard') and self.use_tboard:
            self.tboard_writer.add_scalar(
                '{} Dice'.format('Train' if isTrain else 'Valid'), np.mean(dices), global_step=self.current_epoch)

            self.tboard_writer.add_scalar(
                '{} Chamfer'.format('Train' if isTrain else 'Valid'), np.mean(chamf), global_step=self.current_epoch)

        return np.mean(dices), np.mean(chamf)

    def updater(self):

        for key, (type, scheduler) in self.scheduler.items():
            if type == 'ReduceLROnPlateau':
                scheduler.step(self.monitor[-1])
            elif type == 'CosineAnnealingLR':
                scheduler.step()
            elif type == 'CustomizeLR':
                scheduler.step()
            else:
                assert 'Unknown Scheduler!'

            self.show_outputs(
                "current learning rate: {}".format(
                    self.optimizer[key].state_dict()['param_groups'][0]['lr']))

    def save_er(self):
        if self.save_mode == 'key_epoch':
            if self.best_epoch == self.current_epoch:
                self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-best-checkpoint.pth'.format(self.tag)))
            self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-latest-checkpoint.pth'.format(self.tag)))

        elif self.save_mode == 'all_epoch':
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path = os.path.join(self.cpnt_dire, "Tag-{}-Epoch-{}-cp-{}.pth".format(
                self.tag, self.current_epoch, time_stamp))
            self.save_trainer(path=path)

    def run(self, DEBUG=False):
        self.init_trainer()

        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '-----------------  Run Trainer  ---------------\n'
            '-----------------------------------------------\n'
            'Total epoches:{}\n'
            'Current epoch:{}\n'
            'Criterion: {}\n'
                .format(self.total_epoches, self.current_epoch, self.criterions)
        )
        current_epoch = self.current_epoch

        self.monitor = []
        self.best_epoch = -1
        self.best_index = -1
        self.valider()
        self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-Initial.pth'.format(self.tag)))
        for epoch in range(current_epoch, self.total_epoches):
            self.current_epoch = epoch
            self.trainer()
            if epoch % self.validation_step == 0:
                self.valider()
                self.save_er()
                self.updater()

    def show_outputs(self, info):
        if hasattr(self, 'use_logger') and self.use_logger: logging.info(info)
        if not (hasattr(self, 'use_printt') and not self.use_pprint): print(info)
