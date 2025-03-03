# -*- encoding: utf-8 -*-
"""
Sale Model with Relationships
"""

from datetime import datetime
from apps import db

class Sale(db.Model):
    __tablename__ = 'sales'

    id = db.Column(db.Integer, primary_key=True)
    perfume_id = db.Column(db.Integer, db.ForeignKey('perfumes.id'), nullable=False)
    profession_id = db.Column(db.Integer, db.ForeignKey('professions.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)  # 0 = Female, 1 = Male
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    perfume = db.relationship('Perfume', backref='sales')
    profession = db.relationship('Profession', backref='sales')

    def __init__(self, perfume_id, profession_id, age, gender):
        self.perfume_id = perfume_id
        self.profession_id = profession_id
        self.age = age
        self.gender = gender

    def __repr__(self):
        return f'<Sale {self.id} - Perfume:{self.perfume_id} Profession:{self.profession_id}>'

    @property
    def serialize(self):
        return {
            'id': self.id,
            'perfume': self.perfume.serialize if self.perfume else None,
            'profession': self.profession.serialize if self.profession else None,
            'age': self.age,
            'gender': 'Female' if self.gender == 0 else 'Male',
            'created_at': self.created_at.isoformat()
        }

    @property
    def gender_str(self):
        return 'Female' if self.gender == 0 else 'Male'