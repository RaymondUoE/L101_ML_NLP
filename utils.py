def string_of_list_to_list(string_of_list):
    return string_of_list.strip('[]').replace('\'', '').replace('\"', '').split(', ')