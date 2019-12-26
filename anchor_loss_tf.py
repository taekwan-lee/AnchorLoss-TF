def anchor_loss(onehot_labels, logits, weights, gamma=0.5, slack=0.05):
  """Compute anchor loss between `logits` and the ground truth `labels`.
     Anchor Loss: modulates the standard cross entropy based on the prediction difficulty.
     Loss(x, y) = - y * log(x)
                  - (1 - y) * (1 + x - p_neg)^gamma_neg * log(1-x)
     Args:
       onehot_labels: ground truth one hot labels.
       logits: logits probabilities
       weights: loss weights for each class.(put 1 for no weight.)
       gamma: gamma > 0; reduces the relative loss for well-classiﬁed examples,
              putting more focus on hard, misclassiﬁed examples
       slack: margin variable to penalize the output variables which are close to
              true positive prediction score
       Shape:
            - logits: (N, C) where C is the number of classes
            - onehot_labels: (N, C), should always be onehot, no smoothing allowed..
            - Output: (N, C) same shape as the logits
  """

  with tf.name_scope('anchor_loss'):
    logits = tf.cast(logits, dtype=tf.float32)
    pt = tf.sigmoid(logits)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=onehot_labels, logits=logits)
    labels = tf.argmax(onehot_labels, axis=1)
    class_mask = tf.cast(onehot_labels, dtype=tf.float32)

    gather_idx = tf.concat([tf.cast(tf.range(tf.shape(labels)[0])[:,tf.newaxis], dtype=tf.int64),
                            tf.cast(labels[:,tf.newaxis], dtype=tf.int64)], axis=1)
    pt_pos = tf.gather_nd(pt, gather_idx)
    pt_pos = tf.clip_by_value(pt_pos - slack, clip_value_min=0., clip_value_max=99999.)
    pt_pos = tf.expand_dims(pt_pos, axis=1) # for broadcasting in subtraction
    tf.stop_gradient(pt_pos)

    scaling_neg = tf.pow((1 + tf.subtract(pt, pt_pos)), gamma)
    anchor_loss = class_mask * cross_entropy + (1-class_mask) * scaling_neg * cross_entropy
    weighted_anchor_loss = weights * anchor_loss
    loss = tf.reduce_sum(weighted_anchor_loss, axis=1)
    loss = tf.reduce_mean(loss)
  
  return loss
