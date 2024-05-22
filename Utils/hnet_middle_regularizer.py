# Modification of https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/utils/hnet_regularizer.html#calc_fix_target_reg file
# licensed under the Apache License, Version 2.0, to enable regularization of the interval hypernetwork's output in the middle of interval

import torch

from hypnettorch.hnets import HyperNetInterface

def get_current_targets(task_id, hnet, eps):
    """
    For all j < task_id, compute the output of the hypernetwork. This output
    will be detached from the graph before being added to the return list of
    this function.

    Note, if these targets don't change during training, it would be more memory
    efficient to store the weights θ* of the hypernetwork (which is a fixed
    amount of memory compared to the variable number of tasks). However, it is
    more computationally expensive to recompute h(c_j, θ*) for all j < task_id
    every time the target is needed.

    Note, this function sets the hypernet temporarily in eval mode. No gradients
    are computed.

    See the argument "targets" of calc_fix_target_reg for a use-case of this
    function.

    Parameters:
    -----------

        task_id: int
            The ID of the current task.
        hnet:
            An instance of the hypernetwork before learning a new task (i.e.,
            the hypernetwork has the weights θ* necessary to compute the
            targets).
        eps: float
            A perturbation value.

    Returns:
    --------

        An empty list if task_id is 0. Otherwise, a list of middle targets.
        These targets can be passed to the function calc_fix_target_reg while training on the new task.
    """
    # We temporarily switch to eval mode for target computation (e.g., to get
    # rid of training stochasticities such as dropout).
    hnet_mode = hnet.training
    hnet.eval()

    middle_ret = []

    with torch.no_grad():

        W_middle= hnet.forward(cond_id=list(range(task_id)),
                                                ret_format='sequential',
                                                perturbated_eps=eps,
                                                return_extended_output=False
                                                )
        middle_ret = [[p.detach() for p in W_tid] for W_tid in W_middle]

    hnet.train(mode=hnet_mode)

    return middle_ret

def calc_fix_target_reg(hnet, task_id, eps, middle_targets=None, 
                        mnet=None, prev_theta=None, prev_task_embs=None):
    """
    This regularizer restricts the output-mapping for previous task embeddings.
    For all tasks :math:`j < \text{task\_id}`.

    Parameters:
    ------------

        eps: float
            A perturbation value.
        lower_targets: list, optional
            A list of outputs of the hypernetwork for
            the lower targets. Each list entry must have the output shape as
            returned by the :meth:`hnets.hnet_interface.HyperNetInterface.forward`
            method of the ``hnet``. Note that this function doesn't detach targets.
            If desired, that should be done before calling this function.
        middle_targets: list, optional
            A list of outputs of the hypernetwork for
            the middle targets. Each list entry must have the output shape as
            returned by the :meth:`hnets.hnet_interface.HyperNetInterface.forward`
            method of the ``hnet``.
        upper_targets: list, optional
             A list of outputs of the hypernetwork for
            the upper targets. Each list entry must have the output shape as
            returned by the :meth:`hnets.hnet_interface.HyperNetInterface.forward`
            method of the ``hnet``.
        mnet: 
            Instance of the main network. Has to be provided if
            ``inds_of_out_heads`` are specified.
        prev_theta: list, optional,
            If given, ``prev_task_embs`` but not
            ``targets`` has to be specified. ``prev_theta`` is expected to be
            the internal unconditional weights :math:`\theta` prior to learning
            the current task. Hence, it can be used to compute the targets on
            the fly (which is more memory efficient (constant memory), but more
            computationally demanding).
            The computed targets will be detached from the computational graph.
            Independent of the current hypernet mode, the targets are computed
            in ``eval`` mode.
        prev_task_embs: list, optional
            If given, ``prev_theta`` but not
            ``targets`` has to be specified. ``prev_task_embs`` are the task
            embeddings (conditional parameters) of the hypernetwork.
            See docstring of ``prev_theta`` for more details.

    Returns:
    --------
        The value of the regularizer.
    """
    assert isinstance(hnet, HyperNetInterface)
    assert task_id > 0
    # FIXME We currently assume the hypernet has all parameters internally.
    # Alternatively, we could allow the parameters to be passed to us, that we
    # will then pass to the forward method.
    assert hnet.unconditional_params is not None and \
        len(hnet.unconditional_params) > 0
    assert middle_targets is None or len(middle_targets) == task_id
    assert mnet is not None
    assert middle_targets is None or (prev_theta is None and prev_task_embs is None)
    assert prev_theta is None or prev_task_embs is not None

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))

    # FIXME Assuming all unconditional parameters are internal.
    assert len(hnet.unconditional_params) == \
        len(hnet.unconditional_param_shapes)

    weights = dict()
    uncond_params = hnet.unconditional_params
    weights['uncond_weights'] = uncond_params

    middle_reg = 0

    for i in ids_to_reg:
    
        middle_weights_predicted = hnet.forward(
                                                cond_id=i,
                                                weights=weights,
                                                perturbated_eps=eps,
                                                return_extended_output=False
                                            )


        middle_target = middle_targets[i]
    
        # Regularize all weights of the main network.
        middle_W_target = torch.cat([w.view(-1) for w in middle_target])
        middle_W_predicted = torch.cat([w.view(-1) for w in middle_weights_predicted])
        
        middle_reg_i = (middle_W_target - middle_W_predicted).pow(2).sum()

        middle_reg += middle_reg_i

    return middle_reg / num_regs


if __name__ == '__main__':
    pass