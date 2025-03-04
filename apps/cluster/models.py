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
    
    clusters = db.relationship('Result', backref='cluster', lazy=True)
    
class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    cluster_id = db.Column(db.Integer, db.ForeignKey('clusters.id'), nullable=False)
    sales_id = db.Column(db.Integer, db.ForeignKey('sales.id'), nullable=False)

    def __init__(self, cluster_id, sales_id):
        self.cluster_id = cluster_id
        self.sales_id = sales_id

    def __repr__(self):
        return f'<Result {self.cluster_id} {self.sales_id}>'