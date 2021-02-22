from flask import Blueprint, render_template, request
from .logreg_graphs import *
from .models_code.pytorch_cifar import Logreg, test_dataset, classes
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

file_dir = os.path.dirname(__file__)

# Blueprint Configuration
logreg_bp = Blueprint(
    'logreg_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/logreg/static'
)


@logreg_bp.route('/logreg/concept_logreg', methods=['GET'])
def logreg():
    """Logistic model page."""

    return render_template(
        'index_logreg.html',
        title='logreg main page',
        template='logreg-template',
    )


@logreg_bp.route('/logreg/classification', methods=['GET'])
def logreg_code():
    """Classification with logistic regression page."""

    return render_template(
        'classification.html',
        title='Logreg code page',
        template='logreg-template',
    )


@logreg_bp.route('/logreg/multi_class', methods=['GET', 'POST'])
def logreg_multi():
    """Multi-class logreg classification."""
    if request.method == 'POST':
        idx_to_predict = request.form.to_dict()
        class_label = predict_tch(int(idx_to_predict["num"]))
        return render_template(
            'multi_class.html',
            title='Multi class code page',
            template='logreg-template',
            prediction=class_label,
            index_=int(idx_to_predict["num"])
        )
    return render_template(
        'multi_class.html',
        title='Multi class code page',
        template='logreg-template',
        prediction="None yet",
        index_="None yet (enter a value!)"
    )


def predict_tch(idx_test):
    if idx_test == "":
        return 1
    if idx_test[0] == "0":
        return predict_tch(idx_test[1:])
    idx_test = int(idx_test)
    if idx_test >= 10000:
        return 1
    else:
        path_model = os.path.join(file_dir, "models_code", 'data',
                                  'tch_model.pth')
        path_save = os.path.join(file_dir, 'static')
        model = Logreg(3*32*32, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        checkpoint = torch.load(path_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']  # noqa
        loss = checkpoint['loss']  # noqa
        model.eval()
        img = test_dataset[idx_test][0]
        out = model(img.view(-1, 3*32*32))
        _, predicted = torch.max(out.data, 1)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(path_save, "cifar_img.png"))
        return classes[predicted]


# @logreg_bp.route('/logreg/playing_logreg', methods=['GET'])
# def logreg_play():
#     """Point and click logistic model page."""

#     return render_template(
#         'playing_logereg.html',
#         title='Logreg GUI play page',
#         template='logreg-template',
#     )


# @logreg_bp.route('/logreg/math_logreg', methods=['GET'])
# def logreg_math():
#     """Some maths for the logistic model page."""

#     return render_template(
#         'math_logreg.html',
#         title='logreg math page',
#         template='logreg-template',
#     )
