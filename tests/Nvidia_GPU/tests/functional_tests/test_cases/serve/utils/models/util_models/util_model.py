def fn(input_data):
    res = input_data + "__add_process_fn"
    return res


class ModelC:
    def forward(self, input_data):
        res = input_data + "__add_model_C"
        return res


class ModelD:
    def forward(self, input_data_B, input_data_C):
        output_data = input_data_B + input_data_C
        return output_data
