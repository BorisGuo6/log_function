import copy

GeneralConfig = {}
# ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Breakout']
GeneralConfig['27m_vs_30m']= {
    'name': '27m_vs_30m',
    'min_y': 0,           # Y轴最小值
    'max_y': 1,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Environment Steps', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 1000000,        # x轴主刻度
    'y_major_loc': 0.1,              # y轴主刻度
    'x_minor_loc': 1000000,        # x轴副刻度
    'y_minor_loc': 0.1,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['3s5z_vs_3s6z']= {
    'name': '3s5z_vs_3s6z',
    'min_y': 0,           # Y轴最小值
    'max_y': 0.8,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Environment Steps', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 1000000,        # x轴主刻度
    'y_major_loc': 0.1,              # y轴主刻度
    'x_minor_loc': 1000000,        # x轴副刻度
    'y_minor_loc': 0.1,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['6h_vs_8z']= {
    'name': '6h_vs_8z',
    'min_y': 0,           # Y轴最小值
    'max_y': 0.9,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Environment Steps', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 1000000,        # x轴主刻度
    'y_major_loc': 0.1,              # y轴主刻度
    'x_minor_loc': 1000000,        # x轴副刻度
    'y_minor_loc': 0.1,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['corridor']= {
    'name': 'corridor',
    'min_y': 0,           # Y轴最小值
    'max_y': 1,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Environment Steps', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 1000000,        # x轴主刻度
    'y_major_loc': 0.1,              # y轴主刻度
    'x_minor_loc': 1000000,        # x轴副刻度
    'y_minor_loc': 0.1,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['MMM2']= {
    'name': 'MMM2',
    'min_y': 0,           # Y轴最小值
    'max_y': 1,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Environment Steps', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 1000000,        # x轴主刻度
    'y_major_loc': 0.1,              # y轴主刻度
    'x_minor_loc': 1000000,        # x轴副刻度
    'y_minor_loc': 0.1,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['CartPole_avgreturn']= {
    'name': 'CartPole',
    'min_y': 0,           # Y轴最小值
    'max_y': 500,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Environment Steps', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 500,        # x轴主刻度
    'y_major_loc': 50,              # y轴主刻度
    'x_minor_loc': 100,        # x轴副刻度
    'y_minor_loc': 10,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['CartPole_loss']= {
    'name': 'CartPole',
    'min_y': 0.2,           # Y轴最小值
    'max_y': 0.6,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Loss', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 500,        # x轴主刻度
    'y_major_loc': 0.05,              # y轴主刻度
    'x_minor_loc': 100,        # x轴副刻度
    'y_minor_loc': 0.01,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}
GeneralConfig['CartPole_ratio']= {
    'name': 'CartPole',
    'min_y': 0.56,           # Y轴最小值
    'max_y': 0.58,        # Y轴最大值
    'has_header': False,  #csv文件第一行不是header，那就需要用数字指明col_to_x和col_to_y是第几列的数据
    'col_to_x': 2,        # csv文件中对应的列作为x坐标
    'col_to_y': 3,
    'x_label': 'Ratio', # X轴标题
    'y_label': 'Average Return',    # Y轴标题
    'x_major_loc': 500,        # x轴主刻度
    'y_major_loc': 0.005,              # y轴主刻度
    'x_minor_loc': 100,        # x轴副刻度
    'y_minor_loc': 0.001,               # y轴副刻度
    'pic_color': ['#377eb8', '#e41b1d', '#50b04d', '#c435cc', '#f08536', '#85584e'],
    'pic_width': 7,
    'pic_height': 4,
}

for env, env_config in GeneralConfig.items():
    env_config['fig_title'] = env
    env_config['pic_filename'] = env + '.png'

