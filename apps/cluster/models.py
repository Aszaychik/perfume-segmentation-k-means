# -*- encoding: utf-8 -*-
"""
Cluster Model
"""

from apps import db

class Cluster(db.Model):
    __tablename__ = 'clusters'

    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(80), unique=True, nullable=False)

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f'<Cluster {self.label}>'