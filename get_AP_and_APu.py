'''
读取测试结果文件，输出测试结果

20230502
'''
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file_path_novel', default=None, required=True)

cfgs = parser.parse_args()

class get_AP_and_APu():
    def __init__(self, cfgs):
        self.file_path_Novel = cfgs.file_path_novel

    def print_all_AP_Seen(self):

        APu_dict_Seen = {}
        file = np.load(self.file_path_Seen)
        AP_Seen = np.mean(file[0:30, :, :, :])
        u_list = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2']
        APu_dict_Seen['AP_Seen'] = AP_Seen
        for u_idex, u in enumerate(u_list):
            APu_dict_Seen[type + 'AP_' + u] = np.mean(file[0:30, :, :, u_idex])
        print(APu_dict_Seen)

    def print_all_AP_Similar(self):

        APu_dict_Similar = {}
        file = np.load(self.file_path_Similar)
        AP_Similar = np.mean(file[30:60, :, :, :])
        u_list = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2']
        APu_dict_Similar['AP_Similar'] = AP_Similar
        for u_idex, u in enumerate(u_list):
            APu_dict_Similar[type + 'AP_' + u] = np.mean(file[30:60, :, :, u_idex])
        print(APu_dict_Similar)

    # def print_all_AP_Novel(self):

    #     APu_dict_Novel = {}
    #     file = np.load(self.file_path_Novel)
    #     AP_Novel = np.mean(file[60:90, :, :, :])
    #     u_list = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2']
    #     APu_dict_Novel['AP_Novel'] = AP_Novel
    #     for u_idex, u in enumerate(u_list):
    #         APu_dict_Novel[type + 'AP_' + u] = np.mean(file[60:90, :, :, u_idex])
    #     print(APu_dict_Novel)
    def print_all_AP_Novel(self):

        APu_dict = {}
        file = np.load(self.file_path_Novel)

        AP_Seen = np.mean(file[0:30, :, :, :])
        AP_Similar = np.mean(file[30:60, :, :, :])
        AP_Novel = np.mean(file[60:90, :, :, :])

        test_dataset_type = ['Seen', 'Similar', 'Novel']

        u_list = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2']

        APu_dict['AP_Seen'] = AP_Seen
        APu_dict['AP_Similar'] = AP_Similar
        APu_dict['AP_Novel'] = AP_Novel

        for type in test_dataset_type:
            for u_idex, u in enumerate(u_list):
                if type == 'Seen':
                    APu_dict[type + 'AP_' + u] = np.mean(file[0:30, :, :, u_idex])

                if type == 'Similar':
                    APu_dict[type + 'AP_' + u] = np.mean(file[30:60, :, :, u_idex])

                if type == 'Novel':
                    APu_dict[type + 'AP_' + u] = np.mean(file[60:90, :, :, u_idex])
        print(APu_dict) 


if __name__ == '__main__':
    get_ap = get_AP_and_APu(cfgs)
    get_ap.print_all_AP_Novel()
