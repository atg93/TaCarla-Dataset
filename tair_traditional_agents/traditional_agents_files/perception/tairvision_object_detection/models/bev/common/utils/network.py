
def load_pretrained_weights(pretrained_weights, model_dict):
    if "model" in pretrained_weights:
        pretrained_model_weights = pretrained_weights['model']
    elif "model_state_dict" in pretrained_weights:
        print("Alternative weight is loading...")
        pretrained_model_weights = pretrained_weights['model_state_dict']
    elif "state_dict" in pretrained_weights:
        print("Alternative weight is loading...")
        pretrained_model_weights = pretrained_weights['state_dict']
    else:
        raise ValueError("Convenient weight is not found!")

    for n, p in pretrained_model_weights.items():
        if n.startswith("backbone"):
            n = "encoder." + n
        elif n.startswith("head"):
            n = "head2d." + n
        if not n.startswith("model."):
            n = "model." + n
        if n in model_dict:
            model_weight = model_dict[n]
            if model_weight.shape == p.shape:
                model_dict.update({n: p})
            else:
                # Mismatched shapes
                print(n)

    return model_dict
