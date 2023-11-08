import os
import psutil
import time

def paralleCount():
    count = 0
    pids = psutil.pids() # 获取当前所有进程，返回pid组成的列表
    for i in pids:
        try:
            process = psutil.Process(i) # 获取ID为pids[0]的进程,
            processCommands = process.cmdline()
            if('local_structural_entropy_directed.py' in processCommands):
                count += 1
        except:
            continue
    return count

def main():
    # for beta in range(1,31,3):
    # for embeddingPtr in range(44):
    #     cmd = "python vectorVisualize.py --embeddingPtr {}& ".format(embeddingPtr)
    #     # cmd = "python local_structural_entropy_directed.py --embeddingPtr {}& ".format(embeddingPtr)
    #     os.system(cmd)

    # for beta in range(50,100):
    #     cmd = "python local_structural_entropy_directed.py --beta {}& ".format((beta+1)*0.001)
    #     os.system(cmd)

    for size in range(2,1001):
        cmd = "python local_structural_entropy_directed.py --startCommSize {}& ".format(size)
        os.system(cmd)

        while(paralleCount() > 50):
            time.sleep(10)


if __name__ == "__main__":
    main()
