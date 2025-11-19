from util_models.util_model import fn


class ModelA:
    def forward(self, prompt):
        result = prompt + "__add_model_A"
        return fn(result)


class ModelB:
    def forward(self, input_data):
        res = input_data + "__add_model_B"
        return res


if __name__ == "__main__":
    prompt = "introduce Bruce Lee"
    print(ModelA().forward(prompt))
