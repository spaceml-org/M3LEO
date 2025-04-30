from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from torch import nn


def visualize_attention(dataloader, model):
    for batch in dataloader:
        visualize_with_one_image(batch, model)

        # Break the loop if you only want to extract the first image
        break


def visualize_with_one_image(batch, model):
    # Assuming each batch contains inputs and labels
    inputs, labels = batch

    # Assuming you want to extract the first image from the batch
    batch_image = inputs[0]

    # Convert the image tensor to a NumPy array
    np_image = batch_image.numpy()

    batch_size = np_image.shape[0]

    visualize_original_images(np_image)

    number_of_heads, attention_map = project_images_into_attention_head(
        model, batch_image
    )

    plot_segmentation_images(number_of_heads, attention_map, np_image, batch_size)


def visualize_original_images(np_image):
    _, axs = plt.subplots(4, 4)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(np.moveaxis(np_image[i, :, :, :], 0, 2))

    plt.savefig("figures/multiview.png")


def project_images_into_attention_head(model, image_batch, patch_size=16):
    embeddings_student = model.student_backbone.get_last_selfattention(image_batch)
    return project_images_into_attention_head_embeddings(
        embeddings_student, image_batch, patch_size
    )


def project_images_into_attention_head_embeddings(
    embeddings_last_selfattention, image_batch, patch_size=16
):
    # We want to remove the cls token

    batch_size = embeddings_last_selfattention.shape[0]
    number_of_heads = embeddings_last_selfattention.shape[1]
    attentions = embeddings_last_selfattention[:, :, 0, 1:]

    # number of heads x tokens (input_dimension)

    w_featmap = image_batch.shape[-2] // patch_size
    h_featmap = image_batch.shape[-1] // patch_size

    attentions = attentions.reshape(
        batch_size * number_of_heads, 1, w_featmap, h_featmap
    )
    attentions = (
        nn.functional.interpolate(
            attentions, scale_factor=(patch_size, patch_size), mode="nearest"
        )
        .cpu()
        .detach()
        .numpy()
    )

    attentions = attentions.reshape(
        batch_size, number_of_heads, patch_size * w_featmap, patch_size * h_featmap
    )

    return number_of_heads, attentions


def plot_segmentation_images(
    number_of_heads, attentions_map, original_image, num_of_images_to_log
):
    n_channels = original_image.shape[1]
    batch_size = original_image.shape[0]

    num_of_images_to_log = (
        batch_size if num_of_images_to_log > batch_size else num_of_images_to_log
    )

    fig, axs = plt.subplots(
        num_of_images_to_log, number_of_heads + n_channels, figsize=(30, 18)
    )
    for i in range(num_of_images_to_log):
        for c in range(original_image.shape[1]):
            axs[i, c].imshow(original_image[i, c, :, :].to("cpu").numpy(), cmap="gray")
        for j in range(number_of_heads):
            image = attentions_map[i, j, :, :]
            y_min = np.percentile(image, 1)
            y_max = np.percentile(image, 99)
            image_clip = np.clip(image, y_min, y_max)
            # axs[i, j + n_channels].imshow(image_clip, cmap='jet', vim=y_min, vmax=y_max)
            axs[i, j + n_channels].imshow(image_clip)

            # colorbar.set_ticks([0, 50, 100])  # Replace the values as you need

    # remove the x and y ticks
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.2, hspace=0)
    # plt.savefig("figures/student.png")

    return fig


def plot_attention_map_histograms(
    number_of_heads, attentions_map, original_image, num_of_images_to_log
):
    batch_size = original_image.shape[0]

    num_of_images_to_log = (
        batch_size if num_of_images_to_log > batch_size else num_of_images_to_log
    )

    fig, axs = plt.subplots(
        num_of_images_to_log,
        number_of_heads,
        figsize=(20, 30),
        sharey=True,
        sharex=True,
    )
    max_count = 0
    for i in range(num_of_images_to_log):
        axs[i, 0].set_ylabel("Frequency")
        for j in range(number_of_heads):
            image = attentions_map[i, j, :, :]
            counts, bins = np.histogram(image, bins=10)
            if max(counts) > max_count:
                max_count = max(counts)
            axs[i, j].hist(bins[:-1], bins, weights=counts)
    for j in range(number_of_heads):
        axs[num_of_images_to_log - 1, j].set_xlabel("Attention Map Value")

    plt.ylim(-1, max_count + 10)
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.tight_layout()

    return fig
