import matplotlib.pyplot as plt
from CNN.Network import Network1, Network2, Network3, Network4, Network5, Network6


def plot_loss(loss, title, filename):
    """
    Plots the loss for each epoch.
    :param loss: A list with the loss per epoch.
    :param title: The plot title.
    :param filename: The filename.
    """
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.savefig(f"figures/{filename}.png")


def plot_losses(losses, labels, title, filename):
    """
    Plots multiple losses for each epoch.
    :param losses: A 2D list with the losses per epoch.
    :param labels: The label for each case.
    :param title: The plot title.
    :param filename: The filename.
    """
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for i in range(len(losses)):
        plt.plot(losses[i], label=labels[i])
    plt.legend()
    plt.savefig(f"figures/{filename}.png")


def instantiate_network(architecture):
    """
    Instantiates the network that corresponds to the given
    architecture.
    :param architecture: The network architecture.
    :return: The network.
    """
    if architecture == 1:
        return Network1()
    elif architecture == 2:
        return Network2()
    elif architecture == 3:
        return Network3()
    elif architecture == 4:
        return Network4()
    elif architecture == 5:
        return Network5()
    else:
        return Network6()


def register_hooks(model, architecture, hook):
    """
    Registers the given hook to the given model.
    :param model: The model.
    :param architecture: Model's architecture.
    :param hook: The hook to be registered.
    :return: The model instance and the layers list.
    """
    layers = []
    if architecture == 1:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            }
        ]
    elif architecture == 2:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.conv3.register_forward_hook(hook("conv3"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))
        model.t_conv3.register_forward_hook(hook("t_conv3"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "conv3",
                "channels": model.conv3.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            },
            {
                "name": "t_conv3",
                "channels": model.t_conv3.out_channels
            }
        ]
    elif architecture == 3:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.conv3.register_forward_hook(hook("conv3"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))
        model.t_conv3.register_forward_hook(hook("t_conv3"))
        model.output.register_forward_hook(hook("output"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "conv3",
                "channels": model.conv3.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            },
            {
                "name": "t_conv3",
                "channels": model.t_conv3.out_channels
            },
            {
                "name": "output",
                "channels": model.output.out_channels
            }
        ]
    elif architecture == 4:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.conv3.register_forward_hook(hook("conv3"))
        model.conv4.register_forward_hook(hook("conv4"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))
        model.t_conv3.register_forward_hook(hook("t_conv3"))
        model.t_conv4.register_forward_hook(hook("t_conv4"))
        model.output.register_forward_hook(hook("output"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "conv3",
                "channels": model.conv3.out_channels
            },
            {
                "name": "conv4",
                "channels": model.conv4.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            },
            {
                "name": "t_conv3",
                "channels": model.t_conv3.out_channels
            },
            {
                "name": "t_conv4",
                "channels": model.t_conv4.out_channels
            },
            {
                "name": "output",
                "channels": model.output.out_channels
            }
        ]
    elif architecture == 5:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.conv3.register_forward_hook(hook("conv3"))
        model.conv4.register_forward_hook(hook("conv4"))
        model.conv5.register_forward_hook(hook("conv5"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))
        model.t_conv3.register_forward_hook(hook("t_conv3"))
        model.t_conv4.register_forward_hook(hook("t_conv4"))
        model.output.register_forward_hook(hook("output"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "conv3",
                "channels": model.conv3.out_channels
            },
            {
                "name": "conv4",
                "channels": model.conv4.out_channels
            },
            {
                "name": "conv5",
                "channels": model.conv5.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            },
            {
                "name": "t_conv3",
                "channels": model.t_conv3.out_channels
            },
            {
                "name": "t_conv4",
                "channels": model.t_conv4.out_channels
            },
            {
                "name": "output",
                "channels": model.output.out_channels
            }
        ]
    elif architecture == 6:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.conv3.register_forward_hook(hook("conv3"))
        model.conv4.register_forward_hook(hook("conv4"))
        model.conv5.register_forward_hook(hook("conv5"))
        model.conv6.register_forward_hook(hook("conv6"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))
        model.t_conv3.register_forward_hook(hook("t_conv3"))
        model.t_conv4.register_forward_hook(hook("t_conv4"))
        model.output.register_forward_hook(hook("output"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "conv3",
                "channels": model.conv3.out_channels
            },
            {
                "name": "conv4",
                "channels": model.conv4.out_channels
            },
            {
                "name": "conv5",
                "channels": model.conv5.out_channels
            },
            {
                "name": "conv6",
                "channels": model.conv6.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            },
            {
                "name": "t_conv3",
                "channels": model.t_conv3.out_channels
            },
            {
                "name": "t_conv4",
                "channels": model.t_conv4.out_channels
            },
            {
                "name": "output",
                "channels": model.output.out_channels
            }
        ]

    return model, layers
