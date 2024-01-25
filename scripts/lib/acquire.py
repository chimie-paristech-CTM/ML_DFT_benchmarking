#!/usr/bin/python

def upper_confidence_bound(predictions, variance, beta=2):
    """ Upper Confidence Bound acquisition function"""

    return predictions + beta * variance

