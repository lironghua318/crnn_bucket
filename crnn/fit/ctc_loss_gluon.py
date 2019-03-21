from mxnet import gluon
class CTCLoss(Loss):
    def __init__(self, layout='NTC', label_layout='NT', weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
               "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s"%layout
        assert label_layout in ['NT', 'TN'],\
               "Only 'NT' and 'TN' layouts for label are supported. Got: %s"%label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(CTCLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label,
                       pred_lengths=None, label_lengths=None, sample_weight=None):
        if self._layout == 'NTC':
            pred = F.swapaxes(pred, 0, 1)
        if self._batch_axis == 1:
            label = F.swapaxes(label, 0, 1)
        loss = F.contrib.CTCLoss(pred, label, pred_lengths, label_lengths,
                                 use_data_lengths=pred_lengths is not None,
                                 use_label_lengths=label_lengths is not None,
                                 blank_label='last')
        return _apply_weighting(F, loss, self._weight, sample_weight)