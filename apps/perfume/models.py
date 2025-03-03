# -*- encoding: utf-8 -*-
"""
Perfume Model
"""

from apps import db

class Perfume(db.Model):
    __tablename__ = 'perfumes'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<Perfume {self.name}>'
    
    sales = db.relationship('Sale', backref='perfume')