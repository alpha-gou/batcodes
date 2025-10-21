# 入口文件为my_app.py
file_name=my_app

# 入口文件中的app变量名
entry_variable=my_app

# 线上环境使用
nohup python3 generate_main.py &        # 改写中控服务
nohup python3 rewrite_app_in_new.py &   # 新框架入口服务
nohup python3 chinese_main.py &         # 语文解析服务

# ship环境测试使用
# nohup python3 test_main_ship.py &

# 小数老框架入口服务、语文解析模型接入服务、内部接口服务
gunicorn --reload -c conf/gunicorn.conf $file_name:$entry_variable
