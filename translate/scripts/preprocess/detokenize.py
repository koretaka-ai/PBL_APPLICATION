import os

save_dir = "../../datasets/small_parallel_enja/"
resources_dir = "../../resources/small_parallel_enja/"
if os.path.exists(save_dir)==False:
    os.makedirs(save_dir)


for input, output in [
        (resources_dir+"train.en", save_dir+"train.en"),
        (resources_dir+"dev.en"  , save_dir+"dev.en"),
        (resources_dir+"test.en" , save_dir+"test.en"),
]:
    with open(input, "r") as f:
        data_lines = f.read()

    # 文字列置換
    data_lines = data_lines.replace(" .", ".")
    data_lines = data_lines.replace(" ?", "?")
    data_lines = data_lines.replace(' \'', '\'')
    data_lines = data_lines.replace(" ,", ",")

    with open(output, mode="w") as f:
        f.write(data_lines)

for input, output in [
        (resources_dir+"train.ja", save_dir+"train.ja"),
        (resources_dir+"dev.ja"  , save_dir+"dev.ja"),
        (resources_dir+"test.ja" , save_dir+"test.ja"),
]:
    with open(input, "r") as f:
        data_lines = f.read()

        # 文字列置換
        data_lines = data_lines.replace(" ", "")

        with open(output, mode="w") as f:
            f.write(data_lines)
