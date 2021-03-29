import streamlit as st
from fastai.learner import Learner, load_learner
from fastai.vision.all import PILImage

from ml_for_programmers.cats_vs_dogs.pretrained import is_cat  # noqa
from ml_for_programmers.cats_vs_dogs.pretrained import (
    AnimalType,
    classify,
    model_file_path,
)

learner: Learner = load_learner(model_file_path)


st.title("Cats vs. Dogs")
image_file = st.file_uploader("Image of a cat or dog:", ["png", "jpg", "jpeg"])

if image_file is not None:
    image = PILImage.create(image_file)

    col1, col2 = st.beta_columns((3, 2))
    col1.header("Image")
    col1.image(image, use_column_width=True)

    col2.header("Classification")
    animal, percent = classify(image, learner=learner)
    if animal == AnimalType.cat:
        col2.write(f"... It's a cat! ({percent}%) :cat:")
    else:
        col2.write(f"... It's a dog! ({percent}%) :dog:")
