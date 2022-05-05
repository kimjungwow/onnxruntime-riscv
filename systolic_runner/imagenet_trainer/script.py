import glob
import pandas as pd
import matplotlib.pyplot as plt

# files_txt = "/root/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_trainer/data/trace7s/trace7.log_*"
files_txt = "/root/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_trainer/dir_trace9+/trace9+.log*"


def print_trace_execution(file_name):
    input_size_file = file_name.split("/")[-1].split(".")[0] + "_input_size.csv"
    output_size_file = file_name.split("/")[-1].split(".")[0] + "_output_size.csv"
    total_size_file = file_name.split("/")[-1].split(".")[0] + "_total_size.csv"
    f1 = open(file_name, "r")
    f2 = open(input_size_file, "w")
    f3 = open(output_size_file, "w")
    f4 = open(total_size_file, "w")
    # f2.write("index,is_param,size,shape,elem_size")

    Lines = f1.readlines()
    for line in Lines:
        temp = line.split(" ")
        if len(temp) > 1 and "input[" in temp[1] and temp[1][-1] == "]":
            joined_string = ",".join(temp)
            # f2.write(joined_string)
            f4.write(joined_string)
        if len(temp) > 1 and "output[" in temp[1] and temp[1][-1] == "]":
            joined_string = ",".join(temp)
            # f3.write(joined_string)
            f4.write(joined_string)
    f1.close()
    f2.close()
    f3.close()
    f4.close()


def count_mv_v3():
    FILE_IDX_CONTAINS_NEWLINE = True
    if FILE_IDX_CONTAINS_NEWLINE:
        idx_file = open("file_idx5.txt", "r")

        idx_line = idx_file.readline()
        new_idx_file = open("file_idx5_new.txt", "w")
        while idx_line:
            if len(idx_line.split(" ")) == 3:
                new_idx_file.write(idx_line)

            else:
                new_idx_file.write(idx_line[:-1])
                idx_line = idx_file.readline()
                new_idx_file.write(idx_line)
            idx_line = idx_file.readline()

    idx_file = open("file_idx5_new.txt", "r")
    idx_lines = idx_file.readlines()
    idx = 0
    trace_file = open("trace3+.log", "r")
    line = trace_file.readline()
    idx_line = idx_lines[0]
    while idx < int(idx_line.split(" ")[-1]):
        line = trace_file.readline()
        idx += 1
        # print(idx, " ==== ", line)

    mvin_conv_list = [0, 0, 0, 0, 0]  # bias,input,weight,compute,output
    mvout_conv_list = [0, 0, 0, 0, 0]  # bias,input,weight,compute,output
    layer_type = "null"
    idx_type = "null"
    mvin_csv = open("conv_mvin_trace3_.csv", "w")
    mvout_csv = open("conv_mvout_trace3_.csv", "w")
    mvin_csv.write("layer,bias,input,weight,compute,output,\n")
    mvout_csv.write("layer,bias,input,weight,compute,output,\n")
    for idx_line in idx_lines[1:]:
        if "_" not in idx_line.split(" ")[0]:

            if (any(mvin_conv_list)) or (any(mvout_conv_list)):

                print(layer_type, mvin_conv_list, mvout_conv_list)
                mvin_csv.write(
                    layer_type
                    + ","
                    + str(mvin_conv_list[0])
                    + ","
                    + str(mvin_conv_list[1])
                    + ","
                    + str(mvin_conv_list[2])
                    + ","
                    + str(mvin_conv_list[3])
                    + ","
                    + str(mvin_conv_list[4])
                    + ",\n"
                )
                mvout_csv.write(
                    layer_type
                    + ","
                    + str(mvout_conv_list[0])
                    + ","
                    + str(mvout_conv_list[1])
                    + ","
                    + str(mvout_conv_list[2])
                    + ","
                    + str(mvout_conv_list[3])
                    + ","
                    + str(mvout_conv_list[4])
                    + ",\n"
                )
                mvin_conv_list = [0, 0, 0, 0, 0]
                mvout_conv_list = [0, 0, 0, 0, 0]
                # break
            layer_type = idx_line.split(" ")[0] + "_" + idx_line.split(" ")[1]
            layer_type = idx_line.split(" ")[1]

        mvin_cnt = 0
        mvout_cnt = 0

        while idx < int(idx_line.split(" ")[-1]):
            line = trace_file.readline()
            templine = line.split(",")
            if len(templine) == 9 and templine[1] == "mvin+":
                mvin_cnt += 1
            elif len(templine) == 9 and templine[1] == "mvout+":
                mvout_cnt += 1
            idx += 1
        if idx_type == "conv_mvin_bias":
            mvin_conv_list[0] += mvin_cnt
            mvout_conv_list[0] += mvout_cnt
        elif idx_type == "conv_mvin_input":
            mvin_conv_list[1] += mvin_cnt
            mvout_conv_list[1] += mvout_cnt
        elif idx_type == "conv_mvin_weight":
            mvin_conv_list[2] += mvin_cnt
            mvout_conv_list[2] += mvout_cnt
        elif idx_type == "conv_compute":
            mvin_conv_list[3] += mvin_cnt
            mvout_conv_list[3] += mvout_cnt
        elif idx_type == "conv_mvout_output":
            mvin_conv_list[4] += mvin_cnt
            mvout_conv_list[4] += mvout_cnt
        idx_type = idx_line.split(" ")[1]

        # print(line[:-1], " === ", idx_line[:-1], mvin_cnt, mvout_cnt)
        # if idx > 2000:
        #     break

    # idx = 0

    # while idx_line:
    #     temp = line.split(" ")
    #     if len(temp) == 3:
    #         while idx < int(temp[2]):
    #             line = trace_file.readline()
    #             idx += 1
    #         if temp[2] == "SystolicConvBackpropFilter":
    #             idx_line = idx_file.readline()
    #             while "conv" in idx_line.split(" ")[1]:
    #                 idx_line = idx_file.readline()

    #     else:
    #         idx_line = idx_file.readline()


def inspect_log_v3():
    file_mvin = open("trace3+_mvin.log", "r")
    file_mvout = open("trace3+_mvout.log", "r")
    file_mvin_csv = open("trace3+_mvin.csv", "w")
    file_mvout_csv = open("trace3+_mvout.csv", "w")

    for (file1, txt_name, csv_name) in [
        (file_mvin, "mvin", file_mvin_csv),
        (file_mvout, "mvout", file_mvout_csv),
    ]:
        Lines = file1.readlines()
        idx = 0
        layer_num = "-1"
        layer_type = "null"
        cnt = 0
        # Strips the newline character
        csv_name.write("name,cnt\n")
        for line in Lines:
            temp = line.split(",")
            if len(temp) == 4 and temp[2] == "Kernel_Start":  # and cnt > 0:
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                cnt = 0
                layer_num = str(int(temp[0]))
                layer_type = temp[-1][:-1]
            elif len(temp) >= 5 and "systolicConvTranspose" in temp[2]:
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                cnt = 0
                layer_num = str(int(temp[0]))
                layer_type = "systolicConvTranspose"
            elif len(temp) >= 5 and "SystolicConvBackpropFilter" in temp[2]:
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                cnt = 0
                layer_num = str(int(temp[0]))
                layer_type = "SystolicConvBackpropFilter"
            elif (
                len(temp) >= 4
                and temp[1] == "KJW"
                and temp[2] == "gemmini_loop_conv_ws"
                and temp[3]
                in [
                    "conv_mvin_bias",
                    "conv_mvin_input",
                    "conv_mvin_weight",
                    "conv_compute",
                    "conv_mvout_output",
                ]
            ):
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                    cnt = 0
                    layer_num = temp[0]
            elif (
                len(temp) >= 4
                and temp[1] == "KJW"
                and temp[2] == "gemmini_loop_ws"
                and temp[3]
                in [
                    "move_in_d",
                    "move_in_a",
                    "move_in_b",
                    "compute",
                    "move_out_c",
                ]
            ):
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                    cnt = 0
                    layer_num = temp[0]

            elif len(temp) == 5:
                cnt += 1
        print(txt_name + " Done \n\n")


def parse_large_log_v3():
    file_mvin = open("trace3+_mvin.log", "w")
    file_mvout = open("trace3+_mvout.log", "w")
    glob_list = glob.glob(files_txt)
    glob_list = sorted(glob_list)

    layer_num = 0
    for f in [
        "/root/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_trainer/trace3+.log"
    ]:
        # print(f)
        # continue
        file1 = open(f, "r")
        Lines = file1.readlines()
        idx = 0
        conv_idx = 0
        # Strips the newline character
        for line in Lines:
            temp = line.split(",")
            if (
                len(temp) >= 3
                and temp[0] == "KJW"
                and temp[1] == "gemmini_loop_conv_ws"
                and temp[2]
                in [
                    "conv_mvin_bias",
                    "conv_mvin_input",
                    "conv_mvin_weight",
                    "conv_compute",
                    "conv_mvout_output",
                ]
            ):
                file_mvin.write(str(layer_num) + "_" + str(conv_idx) + "," + line)
                file_mvout.write(str(layer_num) + "_" + str(conv_idx) + "," + line)
                conv_idx += 1
                print(str(layer_num) + "_" + str(conv_idx), temp[2], idx)
            elif (
                len(temp) >= 3
                and temp[0] == "KJW"
                and temp[1] == "gemmini_loop_ws"
                and temp[2]
                in [
                    "move_in_d",
                    "move_in_a",
                    "move_in_b",
                    "compute",
                    "move_out_c",
                ]
            ):
                file_mvin.write(str(layer_num) + "_" + str(conv_idx) + "," + line)
                file_mvout.write(str(layer_num) + "_" + str(conv_idx) + "," + line)
                conv_idx += 1
                print(str(layer_num) + "_" + str(conv_idx), temp[2], idx)
            elif len(temp) >= 3 and temp[0] == "KJW" and temp[1] == "Kernel_Start":
                file_mvin.write(str(layer_num) + "," + line)
                file_mvout.write(str(layer_num) + "," + line)
                layer_num += 1
                conv_idx = 0
                print(layer_num, temp[2], idx)
            elif (
                len(temp) >= 3
                and temp[0] == "KJW"
                and temp[1] == "SystolicConvBackpropFilter++"
            ):
                file_mvin.write(str(layer_num) + ",KJW,SystolicConvBackpropFilter++")
                file_mvout.write(str(layer_num) + ",KJW,SystolicConvBackpropFilter++")
                print(layer_num, "SystolicConvBackpropFilter", idx)
            elif (
                len(temp) >= 3
                and temp[0] == "KJW"
                and temp[1] == "systolicConvTranspose++"
            ):
                file_mvin.write(str(layer_num) + ",KJW,systolicConvTranspose++")
                file_mvout.write(str(layer_num) + ",KJW,systolicConvTranspose++")
                print(layer_num, "systolicConvTranspose", idx)
            elif len(temp) == 9 and temp[0] == "KJW" and temp[1] == "mvin+":
                if temp[6] != "0":  # ignore iszeros == 0
                    cols = str(int(temp[2], 0))
                    rows = str(int(temp[3], 0))
                    accumulator = temp[-2]
                    accumulate = temp[-1]
                    file_mvin.write(
                        cols + "," + rows + "," + accumulator + "," + accumulate + ",-1"
                    )
            elif len(temp) == 9 and temp[0] == "KJW" and temp[1] == "mvout+":
                cols = str(int(temp[2], 0))
                rows = str(int(temp[3], 0))

                pool_stride = temp[-3]
                accumulator = temp[-2]
                is_full = temp[-1]
                file_mvout.write(
                    cols
                    + ","
                    + rows
                    + ","
                    + pool_stride
                    + ","
                    + accumulator
                    + ","
                    + is_full
                )

            idx += 1
        file1.close()
    file_mvin.close()
    file_mvout.close()


def inspect_log_v2():
    file_mvin = open("trace1+_mvin_v2.log", "r")
    file_mvout = open("trace1+_mvout_v2.log", "r")
    file_mvin_csv = open("trace1+_mvin_v2.csv", "w")
    file_mvout_csv = open("trace1+_mvout_v2.csv", "w")

    for (file1, txt_name, csv_name) in [
        (file_mvin, "mvin", file_mvin_csv),
        (file_mvout, "mvout", file_mvout_csv),
    ]:
        Lines = file1.readlines()
        idx = 0
        layer_num = -1
        layer_type = "null"
        cnt = 0
        # Strips the newline character
        csv_name.write("name,cnt\n")
        for line in Lines:
            temp = line.split(",")
            if len(temp) == 4 and temp[2] == "Kernel_Start":  # and cnt > 0:
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                cnt = 0
                layer_num = int(temp[0])
                layer_type = temp[-1][:-1]
            elif len(temp) >= 5 and "systolicConvTranspose" in temp[2]:
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                cnt = 0
                layer_num = int(temp[0])
                layer_type = "systolicConvTranspose"
            elif len(temp) >= 5 and "SystolicConvBackpropFilter" in temp[2]:
                if cnt > 0:
                    csv_name.write(
                        layer_type + "_" + str(layer_num) + "," + str(cnt) + "\n"
                    )
                cnt = 0
                layer_num = int(temp[0])
                layer_type = "SystolicConvBackpropFilter"

            elif len(temp) == 5:
                cnt += 1
        print(txt_name + " Done \n\n")


def parse_large_log_v2():
    file_mvin = open("trace1+_mvin_v2.log", "w")
    file_mvout = open("trace1+_mvout_v2.log", "w")
    glob_list = glob.glob(files_txt)
    glob_list = sorted(glob_list)
    glob_list = [
        "/root/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_trainer/trace1+.log"
    ]

    layer_num = 0
    for f in glob_list:
        # print(f)
        # continue
        file1 = open(f, "r")
        Lines = file1.readlines()
        idx = 0
        # Strips the newline character
        for line in Lines:
            temp = line.split(",")
            if len(temp) >= 3 and temp[0] == "KJW" and temp[1] == "Kernel_Start":
                file_mvin.write(str(layer_num) + "," + line)
                file_mvout.write(str(layer_num) + "," + line)
                layer_num += 1
                print(layer_num, temp[2], f, idx)
            elif (
                len(temp) >= 3
                and temp[0] == "KJW"
                and temp[1] == "SystolicConvBackpropFilter++"
            ):
                file_mvin.write(str(layer_num) + ",KJW,SystolicConvBackpropFilter++")
                file_mvout.write(str(layer_num) + ",KJW,SystolicConvBackpropFilter++")
                print(layer_num, "SystolicConvBackpropFilter", f, idx)
            elif (
                len(temp) >= 3
                and temp[0] == "KJW"
                and temp[1] == "systolicConvTranspose++"
            ):
                file_mvin.write(str(layer_num) + ",KJW,systolicConvTranspose++")
                file_mvout.write(str(layer_num) + ",KJW,systolicConvTranspose++")
                print(layer_num, "systolicConvTranspose", f, idx)
            elif len(temp) == 9 and temp[0] == "KJW" and temp[1] == "mvin+":
                if temp[6] != "0":  # ignore iszeros == 0
                    cols = str(int(temp[2], 0))
                    rows = str(int(temp[3], 0))
                    accumulator = temp[-2]
                    accumulate = temp[-1]
                    file_mvin.write(
                        cols + "," + rows + "," + accumulator + "," + accumulate + ",-1"
                    )
            elif len(temp) == 9 and temp[0] == "KJW" and temp[1] == "mvout+":
                cols = str(int(temp[2], 0))
                rows = str(int(temp[3], 0))

                pool_stride = temp[-3]
                accumulator = temp[-2]
                is_full = temp[-1]
                file_mvout.write(
                    cols
                    + ","
                    + rows
                    + ","
                    + pool_stride
                    + ","
                    + accumulator
                    + ","
                    + is_full
                )

            idx += 1
        file1.close()
    file_mvin.close()
    file_mvout.close()


def parse_large_log():
    file_mvin = open("trace7_mvin.log", "w")
    file_colrow_mvin = open("trace7_colrow_mvin.log", "w")
    file_mvout = open("trace7_mvout.log", "w")
    file_colrow_mvout = open("trace7_colrow_mvout.log", "w")
    file_write_result = open("trace7_write_result.log", "w")
    file_preload = open("trace7_preload.log", "w")
    file_compute_preload = open("trace7_compute_preload.log", "w")
    file_etc = open("trace7_etc.log", "w")
    file_trace = open("trace7_trace.log", "w")

    file_colrow_mvout.write("cols,rows\n")
    file_mvout.write("base_row_addr,dram_addr\n")
    file_colrow_mvin.write("cols,rows\n")
    file_mvin.write("dram_addr,sp_addr\n")
    file_write_result.write("write_result\n")
    file_preload.write("output,preload\n")
    file_compute_preload.write("preload,A,B\n")
    file_trace.write("dram,spm\n")

    for f in glob.glob(files_txt):
        file1 = open(f, "r")
        Lines = file1.readlines()
        idx = 0
        # Strips the newline character
        for line in Lines:
            idx += 1

            # if idx > 100000:
            #     print("KJW,Executed op kernel node")
            #     print(len("KJW,Executed op kernel node"))
            #     sys.exit(0)
            if line[0:27] == "KJW,Executed op kernel node":
                file_mvin.write(line)
                file_colrow_mvin.write(line)
                file_mvout.write(line)
                file_colrow_mvout.write(line)
                file_write_result.write(line)
                file_preload.write(line)
                file_compute_preload.write(line)
                file_etc.write(line)
                file_trace.write(line)
            elif line[0:3] == "KJW":
                temp = line.split(",")
                if len(temp) > 0 and "Called" in temp[-1]:
                    print(temp)
                    x = temp[-1].find("Called")
                    temp[-1] = temp[-1][:x]
                    print(temp)
                elif len(temp) > 0 and "First few output data" in temp[-1]:
                    print(temp)
                    x = temp[-1].find("First few output data")
                    temp[-1] = temp[-1][:x]
                    print(temp)

                try:
                    if len(temp) < 2:
                        file_etc.write(line)

                    elif temp[1] == "mvin" and len(temp) > 5:
                        if temp[4] == "":
                            temp[4] = str(-1)
                        if temp[5] == "":
                            temp[5] = str(-1)
                        file_mvin.write(temp[4] + "," + temp[5] + "\n")
                        file_trace.write(temp[4] + "," + temp[5] + "\n")
                        file_colrow_mvin.write(temp[2] + "," + temp[3] + "\n")
                    elif temp[1] == "mvout" and len(temp) > 5:
                        if temp[4] == "":
                            temp[4] = str(-1)
                        if temp[5] == "":
                            temp[5] = str(-1)
                        file_mvout.write(temp[4] + "," + temp[5] + "\n")
                        file_colrow_mvout.write(temp[2] + "," + temp[3] + "\n")
                        file_trace.write(temp[5] + "," + temp[4] + "\n")
                    elif temp[1] == "write_result" and len(temp) > 2:
                        if temp[2] == "":
                            temp[2] = str(-1)
                        file_write_result.write(temp[2] + "\n")
                        file_trace.write("-1," + temp[2] + "\n")
                    elif temp[1] == "SPM_output_preload_addr" and len(temp) > 3:
                        file_preload.write(temp[2] + "," + temp[3] + "\n")
                    elif temp[1] == "pre_SPMaddrs" and len(temp) > 4:
                        file_compute_preload.write(
                            temp[2] + "," + temp[3] + "," + temp[4] + "\n"
                        )
                    else:
                        file_etc.write(line)
                except Exception as e:
                    print(e)
                    print(f, idx, line, temp)
                    import sys

                    sys.exit(0)
        print(f, " Done, ", str(idx))

        file1.close()
    file_mvin.close()
    file_colrow_mvin.close()
    file_mvout.close()
    file_colrow_mvout.close()
    file_write_result.close()
    file_preload.close()
    file_compute_preload.close()
    file_etc.close()
    file_trace.close()


def plot_trace():
    # df = pd.read_csv("trace7_trace_bak.log", sep=",")
    # df = df.head(1000000)
    # df = df.fillna(-1)

    # df.to_csv("trace7_trace_bak_head_1000000.csv")

    # print(df.shape)
    # print(df["dram"].nunique())
    idx = 1

    df = pd.read_csv("trace7_trace_bak_head_1000000.csv", sep=",")

    print(df)
    print(df.columns)

    fig, ax = plt.subplots(1, 1)
    plt.plot(df.index, df["dram"])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, y2))
    fig.savefig("data/dram_" + str(idx) + ".png")  # , dpi=300)

    fig, ax = plt.subplots(1, 1)
    plt.plot(df.index, df["spm"])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, y2))
    fig.savefig("data/spm_" + str(idx) + ".png")  # , dpi=300)


def plot_mvin():
    df = pd.read_csv("trace7_mvout_parsed.log", sep=",")
    idx = 1
    df = df.fillna(-1)
    # df = df.head(1000000)
    # df.to_csv("trace7_mvout_parsed_1000000.log")

    print(df)
    print(df.columns)

    fig, ax = plt.subplots(1, 1)
    plt.plot(df.index, df["dram_addr"])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, y2))
    fig.savefig("data/mvout/dram_" + str(idx) + ".png")  # , dpi=300)

    fig, ax = plt.subplots(1, 1)
    plt.plot(df.index, df["base_row_addr"])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, y2))
    fig.savefig("data/mvout/spm_" + str(idx) + ".png")  # , dpi=300)


def get_kernels():

    file_name = "0420_1654"
    file1 = open(file_name + ".log", "r")

    line = file1.readline()
    idx = 0
    while line:
        temp = line.split(",")
        if len(temp) == 0:
            continue
        if temp[0] == "KJW" and temp[1] == "Kernel_Start":
            print(idx, temp[2])
            idx += 1
        line = file1.readline()

    file1.close()


def remove_some_prints():

    file_name = "0420_1922"
    file1 = open(file_name + ".log", "r")
    file2 = open(file_name + ".csv", "w")

    line = file1.readline()
    idx = 0
    while line:
        temp = line.split(",")
        if (
            len(temp) == 0
            or temp[0] != "KJW"
            or temp[1] == "gemmini_loop_ws"
            or temp[1] == "gemmini_loop_conv_ws"
        ):
            line = file1.readline()
            continue
        file2.write(line)
        line = file1.readline()

    file1.close()
    file2.close()


def compare_dy():

    file_name = "2_temp_0422_1545"
    file1 = open(file_name + ".log", "r")
    file2 = open(file_name + "_2.log", "w")

    line = file1.readline()
    idx = 0
    testtype = -1
    while line:
        temp = line.split(",")

        if (
            len(temp) > 0
            and temp[0] == "KJW"
            and (
                temp[1] == "SystolicConvBackpropFilter_Print_Start\n"
                or temp[1] == "systolicConvTranspose_Print_Start\n"
            )
        ):
            testtype = 1
        elif (
            len(temp) > 0
            and temp[0] == "KJW"
            and (
                temp[1] == "SystolicConvBackpropFilter_Print_End\n"
                or temp[1] == "systolicConvTranspose_Print_End\n"
            )
        ):
            testtype = -1
        elif testtype == 1:
            line = file1.readline()
            continue
        file2.write(line)
        line = file1.readline()

    file1.close()
    file2.close()


def remove_matrix_prints():

    file_name = "2_temp_0422_1545"
    file1 = open(file_name + ".log", "r")
    file2 = open(file_name + "_2.log", "w")

    line = file1.readline()
    idx = 0
    while line:
        temp = line.split(",")
        if (
            len(temp) == 0
            or temp[0] != "KJW"
            or temp[1] == "gemmini_loop_ws"
            or temp[1] == "gemmini_loop_conv_ws"
        ):
            line = file1.readline()
            continue
        file2.write(line)
        line = file1.readline()

    file1.close()
    file2.close()


if __name__ == "__main__":
    # parse_large_log()
    # plot_trace()
    # plot_mvin()
    # parse_large_log2()
    # get_kernels()
    # remove_some_prints()
    # compare_dy()
    # parse_large_log_v3()
    # inspect_log_v3()
    # count_mv_v3()
    print_trace_execution(
        "/root/chipyard/generators/gemmini/software/onnxruntime-riscv/systolic_runner/imagenet_trainer/trace5+.log"
    )
