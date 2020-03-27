# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:30:57 2019

@author: loktarxiao
"""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class AdaBound(optimizer.Optimizer):
    """Optimizer that implements the AdaBound algorithm.
      See [Lou et al., 2019](https://openreview.net/pdf?id=Bkg3g2R9FX).
      """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=0.001, epsilon=1e-8,
                 amsbound=False, use_locking=False, name='AdaBound'):
        """Construct a new AdaBound optimizer.
        Args:
            learning_rate: A Tensor or a floating point value.  The learning rate.
            beta1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            final_lr: Final learning rate.
            gamma: Convergence speed of the bound functions.
            epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just before
                Section 2.1), not the epsilon in Algorithm 1 of the paper.
            use_locking: If True use locks for update operations.
            amsbound: boolean. Whether to use the AMSBound variant of this algorithm.
            name: Optional name for the operations created when applying gradients.
                Defaults to "AdaBound".
        """
        super(AdaBound, self).__init__(use_locking, name)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._final_lr = final_lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._amsbound = amsbound
        self._base_lr = learning_rate

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._final_lr_t = None
        self._gamma_t = None
        self._epsilon_t = None
        self._base_lr_t = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._beta1,
                                       name="beta1_power",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2,
                                       name="beta2_power",
                                       colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        final_lr = self._call_if_callable(self._final_lr)
        gamma = self._call_if_callable(self._gamma)
        epsilon = self._call_if_callable(self._epsilon)
        base_lr = self._call_if_callable(self._base_lr)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._final_lr_t = ops.convert_to_tensor(final_lr, name="final_learning_rate")
        self._gamma_t = ops.convert_to_tensor(gamma, name="gamma")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name='epsilon')
        self._base_lr_t = ops.convert_to_tensor(base_lr, name="base_learning_rate")

    def _apply_dense(self, grad, var):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        final_lr_t = math_ops.cast(self._final_lr_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = final_lr_t * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_t + 1.))
        upper_bound = final_lr * (1. - 1. / (gamma_t))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t + m_scaled_g_values, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t + v_scaled_g_values, use_locking=self._use_locking)
        vhat = self.get_slot(var, "vhat")
        if self._amsbound:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(vhat, v_t))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var,
                                          m_t * math_ops.clip_by_value(lr / (v_sqrt + epsilon_t), lower_bound,
                                                                       upper_bound),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, grad.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        final_lr_t = math_ops.cast(self._final_lr_t, grad.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, grad.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = final_lr_t * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_t + 1.))
        upper_bound = final_lr * (1. - 1. / (gamma_t))
        var = var.handle
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m").handle
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t + m_scaled_g_values, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v").handle
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t + v_scaled_g_values, use_locking=self._use_locking)
        vhat = self.get_slot(var, "vhat").handle
        if self._amsbound:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(vhat, v_t))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var,
                                          m_t * math_ops.clip_by_value(lr / (v_sqrt + epsilon_t), lower_bound,
                                                                       upper_bound),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        final_lr_t = math_ops.cast(self._final_lr_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = final_lr_t * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_t + 1.))  # *t
        upper_bound = final_lr * (1. - 1. / (gamma_t))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        vhat = self.get_slot(var, "vhat")
        if self._amsbound:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(vhat, v_t))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        var_update = state_ops.assign_sub(var,
                                          m_t * math_ops.clip_by_value(lr / (v_sqrt + epsilon_t), lower_bound,
                                                                       upper_bound),
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(
                    x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)

import math
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    initial_lrate = 0.05
    drop = 0.5
    epochs_drop = 10.0
    if epoch <= 4:
        lr = 0.01 * epoch + initial_lrate
    else:
        lr = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    print('Learning rate: ', lr)
    return lr
