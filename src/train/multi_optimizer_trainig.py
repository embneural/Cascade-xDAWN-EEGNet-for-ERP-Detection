from .misc import Training
import numpy as np
from ..models.dev.visualize_sinc_filters import visulize_filter, plot_hist

class Multi_optimizer_Training(Training):
    
    def early_stop(self,data_loader:dict, fs):

        """ 
        using early stop to train
        """
        for self.epoch in range(self.max_epoch):

            '''
            train
            '''
            plot_hist(self.vis, self.model)
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

            title = 'Acc'
            self.vis.line(
                    X = np.column_stack((
                            self.epoch,
                            self.epoch,
                        )),
                    Y = np.column_stack((
                            np.array(train_acc),
                            np.array(self.valid_acc),
                        )),
                    
                    opts = { 
                            'title'   :  title,
                            'legend'  :  ['train', 'valid'],
                            },
                    
                    win = title,
                    update='append',
                )
            
            title = 'marcro recall'
            self.vis.line(
                     X = np.column_stack((
                             self.epoch,
                             self.epoch,
                        )),
                     Y = np.column_stack((
                             np.array(train_recall).mean(),
                             np.array(self.valid_recall).mean(),
                             
                         )),
                     
                     opts = { 
                            'title'   :  title,
                            'legend'  :  ['train', 'valid'],
                              # 'dash': np.array(['solid', 'dash']),
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
            
            visulize_filter(self.vis, self.model.filterbank.sinc.sinc_filters.cpu().detach().numpy()[:, 0, :], fs = fs)


            
            
            
            
            
            