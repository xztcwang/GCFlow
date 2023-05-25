"""Basic definitions for the flows module."""

#import utils
import flow_ssl.nsf_utils as utils
#from nde import distributions
from flow_ssl.nde import distributions
import copy

class Flow(distributions.Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, x_adj, context=None):
        #noise, logabsdet = self._transform(inputs, context=context)
        #log_prob = self._distribution.log_prob(noise, context=context)
        #return log_prob + logabsdet

        z, logabsdet, = self._transform(x_adj, context=context)
        return z, logabsdet

        # if isinstance(x_adj, tuple):
        #     x = x_adj[0]
        #     adj = x_adj[1]
        #     if self.flowtype == 'gat':
        #         if self.static:
        #             attention = self._att_transfrom(self.x_orig, self.adj_orig)
        #             attention = attention.to_sparse()
        #         else:
        #             attention = self._att_transfrom(x, self.adj_orig)
        #             attention = attention.to_sparse()
        #         #attention = attention.detach()
        #         x_adj = (x, attention)
        #         z, logabsdet, attention = self._transform(x_adj, context=context)
        #         return z, logabsdet, attention
        # else:
        #     print("ERROR")

    def _sample(self, num_samples, context):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = utils.merge_leading_dims(noise, num_dims=2)
            context = utils.repeat_rows(context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = utils.merge_leading_dims(noise, num_dims=2)
            context = utils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = utils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=context)
        return noise
