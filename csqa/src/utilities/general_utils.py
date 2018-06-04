
def split_list_in_chunks(input_list, num_chunks):
    return [input_list[i::num_chunks] for i in range(num_chunks)]