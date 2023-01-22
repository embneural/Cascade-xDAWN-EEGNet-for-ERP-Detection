

import os, sys
import numpy as np
import mne
from collections import Counter
import scipy
import torch
from ...train.misc import Training
# from ...models.dev.visualize_sinc_filters import visulize_filter
# from ...models.dev.visualize_sinc_filters import plot_hist

ch_names =  [   'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5',
                'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',  'CP5', 'CP3',
                'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2',
                'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7',  'F5',  'F3',
                'F1',  'Fz',  'F2',  'F4',  'F6',  'F8',  'FT7', 'FT8',
                'T7',  'T8',  'T9',  'T10', 'TP7', 'TP8', 'P7',  'P5',
                'P3',  'P1',  'Pz',  'P2',  'P4',  'P6',  'P8',  'PO7',
                'PO3', 'POz', 'PO4', 'PO8', 'O1',  'Oz',  'O2',  'Iz',  'stim']
eeg_ch_n = 64
sfreq = 240
ch_types = ['eeg']*eeg_ch_n + ['stim']
event_dict = {'non': 1, 'target': 2}
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
info.set_montage('standard_1020')


def code_transfer(test_data_code, label):
    test_data_code = np.squeeze(test_data_code, axis = 0)
    new_code = test_data_code[test_data_code != 0]
    new_code = new_code.reshape(len(label), int(new_code.shape[0]/len(label)))
    stimul_code = new_code[:, ::24]
    return stimul_code

def data_transfer(test_data, label):    
    a, b, c, d = test_data.shape
    test_data = test_data.reshape(len(label), 180, b, c, d)
    test_data = torch.tensor(test_data, dtype = torch.float32)
    return test_data

def get_data_IIb(mode:str, path: str):
    '''
    load BCI competition II P300 data
    mode: str:  'train' or 'test'
        load which data
    path: str
        original data path
    Returns
	-------
    '''
    if mode == 'train':
        label = 'train_label.txt'
        s = 1
        e = 11
    elif mode == 'test':
        label = 'test_label.txt'
        s = 1
        e = 8
    first = True
    for i in range(s, e+1):
        if mode == 'train':
            if i <=5:
                name = 'AAS010R0'+ str(i)+'.mat'
            else:
                name = 'AAS011R0'+ str(i-5)+'.mat'
        elif mode == 'test':
            name = 'AAS012R0' + str(i) + '.mat'
        print(name)
        file = os.path.join(path,name)
        x = scipy.io.loadmat(file)
        Signal = x['signal'] #闪烁频率，可以从中提取出 after stimulate  none target's  response
        Flashing = x['Flashing']
        StimulusCode = x['StimulusCode']
        Signal = Signal.transpose(1,0)  # 0~1
        Flashing = Flashing.transpose(1,0)
        StimulusCode = StimulusCode.transpose(1,0)  
        if first:
            tSignal = Signal
            tFlashing = Flashing
            tStimulusCode = StimulusCode
            first = False
        else:
            tSignal       = np.concatenate((tSignal,Signal), axis = -1)
            tFlashing     = np.concatenate((tFlashing,Flashing), axis = -1)
            tStimulusCode = np.concatenate((tStimulusCode,StimulusCode), axis = -1)
    '''
    读取标签
    '''
    with open(os.path.join(path, label), "r") as f:  # 打开文件
        Target = f.read()  # 读取文件
    '''
    获得block的开始id
    '''
    intense_id = np.where(tFlashing[0] == 1)[0]
    block_s_list = [intense_id[0]]
    for i in range(len(intense_id)-1):
        if (intense_id[i+1] - intense_id[i]) > (2.0*240):
            block_s = intense_id[i+1]
            block_s_list.append(block_s)
    block_s_list.append(len(tFlashing[0]))
    block_n = len(block_s_list) - 1
    chara_map = {'A':(7,1), 'B':(7,2), 'C':(7,3), 'D':(7,4), 'E':(7,5), 'F':(7,6),
                    'G':(8,1), 'H':(8,2), 'I':(8,3), 'J':(8,4), 'K':(8,5), 'L':(8,6),
                    'M':(9,1), 'N':(9,2), 'O':(9,3), 'P':(9,4), 'Q':(9,5), 'R':(9,6),
                    'S':(10,1), 'T':(10,2), 'U':(10,3), 'V':(10,4), 'W':(10,5), 'X':(10,6),
                    'Y':(11,1), 'Z':(11,2), '1':(11,3), '2':(11,4), '3':(11,5), '4':(11,6),
                    '5':(12,1), '6':(12,2), '7':(12,3), '8':(12,4), '9':(12,5), '_':(12,6) }

    t_trigger = np.zeros(tFlashing.shape[1])    
    enable_in = True
    for k in range(block_n):
        s = block_s_list[k]
        e = block_s_list[k+1]
        
        signal = tSignal[:, s:e ]
        StimulusCode = tStimulusCode[:,s:e]
        Flashing = tFlashing[:,s:e]
        trigger = np.zeros(Flashing.shape[1])
        '''
        构建trigger
        ''' 
        print(chara_map[Target[k]])
        a, b = chara_map[Target[k]] # k [0, 41]  
        index_a  = np.where( StimulusCode[0] == a )
        index_b  = np.where( StimulusCode[0] == b )
        id_ = np.where(Flashing[0] == 1)  # 包含了non和p300的id
        p300_id = np.union1d(index_a, index_b) # 取并集
        non_id = np.setdiff1d(id_, p300_id)  # 取差集
        
        trigger[non_id] = 1
        trigger[p300_id] = 2
        '''
        更改  t_trigger
        '''
        t_trigger[s:e] = trigger
    '''
    只取第一个
    '''
    non_id  = np.where(t_trigger == 1)[0]
    p300_id = np.where(t_trigger == 2)[0]
    a = np.setdiff1d(  non_id  ,  non_id[::24] )
    b = np.setdiff1d(  p300_id , p300_id[::24] )
    c = np.union1d(a,b)
    t_trigger[c] = 0
    print(Counter(t_trigger))
    print(Counter(t_trigger)[1]/Counter(t_trigger)[2])
    # x1 = np.arange(t_trigger.shape[0])
    # plt.scatter(x1, t_trigger)
    x = np.concatenate((tSignal, t_trigger[np.newaxis, :]), axis = 0)

    return mne.io.RawArray(x, info), tStimulusCode, Target







def get_data_III(subject:str, mode, path):
    '''
    load BCI competition III P300 data
    subject: str: 'A' or 'B'
    mode: str:  'train' or 'test'
        load which data
    path: str
        original data path
    Returns
	-------
    '''
    if subject not in['A', 'B']:
        raise ValueError("wrong subject")
    
    if mode not in ['train', 'test']:
        raise ValueError("wrong mode")
        
    if mode == 'train':
        name = 'Subject_'+ subject +'_Train.mat'

    elif mode == 'test':
        name  = 'Subject_' + subject +'_Test.mat'
        label = 'Subject_'+ subject +'_Test_label.txt'
        
    file = os.path.join(path, name)
    x = scipy.io.loadmat(file)
    Signal = x['Signal'] #闪烁频率，可以从中提取出 after stimulate  none target's  response
    
    Flashing = x['Flashing']
    StimulusCode = x['StimulusCode'] #没什么用

    StimulusCode = np.expand_dims(StimulusCode, axis = -1)
    Flashing     = np.expand_dims(Flashing, axis = -1)

    Signal = Signal.transpose(0, 2, 1)
    Flashing = Flashing.transpose(0, 2, 1)
    StimulusCode = StimulusCode.transpose(0, 2, 1)
    
    if mode == 'train':
        Target = str(x['TargetChar'][0])
    
    elif mode == 'test':
        with open(os.path.join(path, label), "r") as f:  # 打开文件
            Target = f.read()  # 读取文件

    chara_map = {'A':(7,1), 'B':(7,2), 'C':(7,3), 'D':(7,4), 'E':(7,5), 'F':(7,6),
                    'G':(8,1), 'H':(8,2), 'I':(8,3), 'J':(8,4), 'K':(8,5), 'L':(8,6),
                    'M':(9,1), 'N':(9,2), 'O':(9,3), 'P':(9,4), 'Q':(9,5), 'R':(9,6),
                    'S':(10,1), 'T':(10,2), 'U':(10,3), 'V':(10,4), 'W':(10,5), 'X':(10,6),
                    'Y':(11,1), 'Z':(11,2), '1':(11,3), '2':(11,4), '3':(11,5), '4':(11,6),
                    '5':(12,1), '6':(12,2), '7':(12,3), '8':(12,4), '9':(12,5), '_':(12,6) }
    '''
    k : 训练集[0, 85)   ， 测试集 k : [ 0, 100 )
    '''
    if mode == 'train':
        char_num = 85
    elif mode == 'test':
        char_num = 100

    stimulate = np.zeros([char_num, 1, 7794])    
    for k in range(0, char_num):  
        '''  重构 flashing  '''  
        a, b = chara_map[Target[k]]
        
        index_a  = np.where( StimulusCode[k][0] == a )
        index_b  = np.where( StimulusCode[k][0] == b )
        
        Flashing[k][0][index_a] = 2
        Flashing[k][0][index_b] = 2
        '''  从flashing中提取出第一个元素，即trigger  '''    
        non_index    =  np.where(Flashing[k][0] == 1 )[0][::24]
        target_index =  np.where(Flashing[k][0] == 2 )[0][::24]
        
        stimulate[k][0][non_index]    = 1
        stimulate[k][0][target_index] = 2
    '''拼接创建raw数据'''
    a,b,c = Signal.shape
    x = np.zeros([a, b+1, c])
    x[:, 0:64, :] = Signal
    x[:, -1,   :] = np.squeeze ( stimulate, axis = 1)
    
    x = x.transpose(1, 0, 2)
    x = x.reshape(65, -1)

    return mne.io.RawArray(x, info), StimulusCode.transpose(1, 0, 2), Target


def preprocess(epoch_len: float,detrend, baseline, my_filter, downsample: int, raw, plot = False):
    '''
    Preprocessing
    my_filter: ['mne', l_freq, h_freq] or ['my', filter] or [None] 
    Returns
    ----
    data_x
    data_y
    '''
    raw = raw.pick_types(eeg=True, stim=True, ecg = False, eog=False)
    
    if my_filter[0] == 'mne':         # ['mne', l_freq, h_freq]
        l_freq = my_filter[1]
        h_freq = my_filter[2]
        raw.filter(l_freq, h_freq)
    
    '''
    创建epoch数据
    '''
    events = mne.find_events(raw, stim_channel='stim', initial_event = True)
    epochs = mne.Epochs(raw, events, event_id = event_dict, tmin = 0, tmax = epoch_len, 
                baseline = baseline, detrend = detrend, picks = ['eeg'], preload = True, verbose = None)
        
    if plot: # plot average epochs  
        plot_picks = ['eeg']
        evo_kwargs = dict(picks=plot_picks, spatial_colors=True,
                            verbose='error')  # ignore warnings about spatial colors
        
        for key in event_dict.keys():    
            fig = epochs[key].average(picks=plot_picks).plot(**evo_kwargs)
            fig.suptitle('detrend: {}, baseline: {}, {} epochs average '.format(detrend, baseline, key))
            mne.viz.tight_layout()
    '''
    remove the last time sample point, filter data, downsample
    '''
    if my_filter[0] == None or my_filter[0] == 'mne':
        data_x = epochs.get_data()[:,:,:-1][:,:,::downsample]
    
    elif my_filter[0] == 'my':         # ['my', filter]
        filter = my_filter[1]
        data_x = filter.apply (epochs.get_data()[:,:,:-1]) [:,:,::downsample]        
    
    data_y = epochs.events[:, -1]
    
    return [data_x, data_y-1]


class BCI_II_III_Training(Training):
    def early_stop(self,data_loader:dict, recognize_data):
        """ 
        using early stop to train
        """
        if not hasattr(self, 'updatetextwindow'):
            self.updatetextwindow = self.vis.text('number of correctly recognized symobols')
        
        for self.epoch in range(self.max_epoch):      
            '''
            train
            '''
            train_loss, train_y, train_pred_y = self.train(self.model, data_loader['train'])
            '''
            scheduler
            '''
            if self.epoch > self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                # Update bn statistics for the swa_model at the end
                '''
                update bn every time when the swa model' weihts update, maybe this way is good
                '''
                self.update_bn(data_loader['train'], self.swa_model, device = self.device)
                self.got_model = self.swa_model 
            else:
                for sch in self.scheduler:
                    sch.step()
                self.got_model = self.model
            '''
            valid
            '''
            self.valid_loss, valid_y, valid_pred_y  = self.evaluate(self.got_model, data_loader['valid'])
            '''
            compute performance
            '''
            train_recall = self.compute_recall(self.event_dict, train_y, train_pred_y)
            train_acc    = (train_y == train_pred_y).sum()/ len(train_y)
            self.valid_recall = self.compute_recall(self.event_dict, valid_y, valid_pred_y)
            self.valid_acc    = (valid_y == valid_pred_y).sum()/ len(valid_y)
            
            self.get_report(valid_y, valid_pred_y)
               
            '''
            patience
            '''
            if self.reach_patience(self.early_type):
                break
            '''
            save
            '''
            if self.save_path != None:
                self.pth_save(self.save_path)
            '''
            dynamic plot
            '''
            title = 'loss'
            self.vis.line(
                     X = np.column_stack((
                             self.epoch,
                             self.epoch,
                        )),
                     Y = np.column_stack((
                             train_loss,
                             self.valid_loss,
                         )),
                     opts = { 
                            'title'   :  title,
                            'legend'  :  ['train', 'valid'],
                             'dash': np.array(['solid', 'solid']),
                             },
                     win = title,
                     update='append',
                 )

            title = 'performance'
            self.vis.line(
                    X = np.column_stack((
                            self.epoch,
                            self.epoch,
                        )),
                    Y = np.column_stack((
                            np.array(self.report['accuracy']),
                            np.array(self.report['1']['f1-score']),
                        )),
                    
                    opts = { 
                            'title'   :  title,
                            'legend'  :  ['valid_acc', 'valid_f1_score'],
                            },
                    
                    win = title,
                    update='append',
                )

            title = 'train recall'
            self.vis.line(
                    X = np.array([self.epoch] * len(self.event_dict.keys() ) ).T [None, :]  , 
                    Y = np.array(train_recall).T [None, :],  
                    opts = { 
                            'title'   :  title,
                            'legend'  :  [key for key in self.event_dict.keys()],
                            },
                    win = title,
                    update='append',
                )
            
            title = 'valid recall'
            self.vis.line(
                    X = np.array([self.epoch] * len(self.event_dict.keys() ) ).T [None, :]  , 
                    Y = np.array(self.valid_recall).T [None, :],  
                    opts = { 
                            'title'   :  title,
                            'legend'  :  [key for key in self.event_dict.keys()],
                            },
                    win = title,
                    update='append',
                )
            
            symbol_list, results = self.recognize_with_less_time(self.got_model, recognize_data[0], recognize_data[1], recognize_data[2], max_repetition = 15)
            self.vis.text(str(self.epoch) + ':' + str(results), win= self.updatetextwindow, append=True)

            title = 'Sum'        
            self.vis.line(
                    X = np.column_stack((
                        
                            [self.epoch for i in range(len(symbol_list))]
                        
                        )),
                    Y = np.column_stack((                            
                            [sum(i)  for i in symbol_list]
    
                        )),
                    opts = { 
                            'title'   :  title,
                            'legend'  :  [ str(i) for i in self._range],
                            },
                    
                    win = title,
                    update='append',
                )

    def recognize_with_less_time(self, model, test_data, stimul_code, label, max_repetition = 15):
        # 进行识别
        from scipy.special import softmax
        model.eval()
        with torch.no_grad():   
            number = test_data.shape[0]
            r_stack = None
            for index in range(number):
                X = test_data[index].to(self.device)
                pred = model(X)
                result = pred.cpu().detach().numpy()[None, :, :]
                
                if index == 0:
                    r_stack = result
                    continue
                r_stack = np.concatenate((r_stack, result), axis = 0)

            symbol_list = []
            self._range = ['logits', 1, 10] # temperature
            for t in self._range:
                if t != 'logits':
                    output = softmax(r_stack/t,axis = -1)
                else:
                    output = r_stack
                    
                correct_recgonize = self.get_recognized_symbols(max_repetition, number, output , stimul_code, label)
                symbol_list.append(correct_recgonize)
            
            max_idx = np.array(2)  # with a higher temperature or just adopt logits, the symbol acc will be a liiter more stable
            results = symbol_list[max_idx]
            
            return symbol_list, results
            # print(results)

    def get_recognized_symbols(self, max_repetition, number, r_stack, stimul_code, label):
        correct_recgonize = []
        for iteration in range(1, max_repetition+1):
            count = 0
            for index in range(number):
                result = r_stack[index]
                '''
                进行预测
                '''
                total_p_r = 0
                total_p_c = 0
                for iter in range(1, iteration+1):
                
                    s = 12*(iter - 1)
                    e = 12*iter
                    
                    a = stimul_code[index,  s:e]
                    b = result[s:e][:,1]
                    
                    p_r = []  # 行 7~12 的概率
                    p_c = []  # 列 1~6  的概率
                    
                    for i in range(7,13):    
                        
                        p_r.append ( b[ np.where(a==i)[0][0] ] )
                        
                    for i in range(1, 7):
                        p_c.append ( b[ np.where(a==i)[0][0] ] )
                    
                    
                    p_r = np.array(p_r)
                    p_c = np.array(p_c)   # 每一个iter的 概率list
                    
                    total_p_r += p_r
                    total_p_c += p_c
                    
                r = np.argmax(total_p_r)+7   # r的值为 7 ~ 12
                c = np.argmax(total_p_c)+1   # c的值为 1 ~ 6
                chara_map = np.array(  ['A', 'B', 'C', 'D', 'E', 'F',
                                        'G', 'H', 'I', 'J', 'K', 'L',
                                        'M', 'N', 'O', 'P', 'Q', 'R',
                                        'S', 'T', 'U', 'V', 'W', 'X',
                                        'Y', 'Z', '1', '2', '3', '4',
                                        '5', '6', '7', '8', '9', '_' ])
                
                chara_map = chara_map.reshape(6,6)
                pred = chara_map[r-7, c-1]
                
                # print(pred, end ="")
                
                if label[index] == pred:
                    count += 1
            # if verobose:
            #     print(count)
            correct_recgonize.append(count)
            
        return correct_recgonize
    
  