import configparser
import argparse
import logging
import os
import warnings
import torch
from fl import FL


# def read_config():
#     config = configparser.ConfigParser()
#     config.read('/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_A_test_B') # FIRST TEST
#     return config
 
# config = read_config()

# fl = FL(config)

# fl.start()


path = ['/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/opp/dccae/A30_B0_AB0_label_A_test_A',

'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A10_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_gyro/A30_B0_AB0_label_A_test_A',


'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A10_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/acce_mage/A30_B0_AB0_label_A_test_A',



'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A10_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/mhealth/split_ae/gyro_mage/A30_B0_AB0_label_A_test_A',

'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A10_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_depth/A30_B0_AB0_label_A_test_A',

'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A10_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/acce_rgb/A30_B0_AB0_label_A_test_A',



'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_AB_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A0_B30_AB0_label_B_test_B',


'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A10_B0_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A10_B0_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A10_B10_AB30_label_A_test_B',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A10_B10_AB30_label_B_test_A',
'/Users/lc/desktop/iotdi22-mmfl/config/ur_fall/split_ae/rgb_depth/A30_B0_AB0_label_A_test_A',  
]

i = 0
path_listCOmpleted=[]
for thePath in path:
    def read_config():
        config = configparser.ConfigParser()
        config.read(thePath) # FIRST TEST
        print("doing" + thePath)
        path_listCOmpleted.append(thePath)
        print(path_listCOmpleted)
        print(i)

        return config
    
    config = read_config()

    fl = FL(config)

    fl.start()