# 杀死全部GPU上的进程
ps -ef | grep 'run' | grep -v grep | awk '{print $2}' | xargs kill