# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.cluster import blueprint
from apps.cluster.models import Cluster, Result
from apps import db
from apps.sale.models import Sale
from apps.perfume.models import Perfume
from apps.profession.models import Profession
from flask import render_template, request, redirect, current_app
from flask_login import login_required
from jinja2 import TemplateNotFound
import os
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sqlalchemy import create_engine

def perform_kmeans(iterations, variables, centroid_ids=[5,10,15,20,25]):
    # Fetch sales data with perfume and profession names
    query = db.session.query(
        Sale.id,
        Sale.age,
        Sale.gender,
        Sale.profession_id,
        Sale.perfume_id,
        Perfume.name.label('perfume_name'),
        Profession.name.label('profession_name')
    ).join(Perfume, Sale.perfume_id == Perfume.id
    ).join(Profession, Sale.profession_id == Profession.id)
    
    sales_data = query.all()
    
    if not sales_data:
        raise ValueError("No sales data available for clustering.")
    
    # Convert sales data to DataFrame
    df_sales = pd.DataFrame(sales_data, columns=[
        'id', 'age', 'gender', 'profession_id', 'perfume_id', 
        'perfume_name', 'profession_name'
    ])

    # Extract initial centroids in specified order
    initial_centroids = []
    for cid in centroid_ids:
        centroid_data = df_sales[df_sales['id'] == cid][variables]
        if centroid_data.empty:
            raise ValueError(f"Centroid ID {cid} not found in sales data.")
        initial_centroids.append(centroid_data.values[0])
    centroids = np.array(initial_centroids)

    # Prepare feature matrix
    X = df_sales[variables].values

    # Custom K-Means algorithm
    for _ in range(iterations):
        # Calculate |Σ(x_i - c_i)| distances
        raw_diffs = (X[:, None, :] - centroids[None, :, :]).sum(axis=2)
        dists = np.abs(raw_diffs)
        
        # Assign to nearest cluster (0-indexed)
        labels = np.argmin(dists, axis=1)
        df_sales['cluster'] = labels
        
        # Update centroids (handle empty clusters)
        new_centroids = np.zeros_like(centroids)
        for j in range(len(centroid_ids)):
            cluster_data = df_sales[df_sales['cluster'] == j]
            if not cluster_data.empty:
                new_centroids[j] = cluster_data[variables].mean().values
            else:
                new_centroids[j] = centroids[j]  # Keep previous centroid
        
        centroids = new_centroids

    # Create cluster summary dataframes
    # Full precision numerical means
    df_sales_with_labels = df_sales.groupby('cluster').agg({
        'age': 'mean',
        'perfume_id': 'mean',
        'gender': 'mean',
        'profession_id': 'mean',
    }).reset_index()
    
    # Rounded numerical means
    df_sales_with_labels_rounded = df_sales_with_labels.copy()
    for col in ['age', 'perfume_id', 'gender', 'profession_id']:
        df_sales_with_labels_rounded[col] = df_sales_with_labels_rounded[col].round(0).astype(int)
    
    # Rounded with categorical modes
    df_sales_with_labels_rounded_name = df_sales_with_labels_rounded.copy()
    
    # Get mode for perfume and profession names
    name_modes = df_sales.groupby('cluster').agg({
        'perfume_name': lambda x: x.mode()[0] if not x.empty else None,
        'profession_name': lambda x: x.mode()[0] if not x.empty else None
    }).reset_index()
    
    # Merge name modes with numerical data
    df_sales_with_labels_rounded_name = df_sales_with_labels_rounded_name.merge(
        name_modes, on='cluster', how='left'
    )

    # Convert to list of dictionaries for response
    full_precision = df_sales_with_labels.apply(lambda row: row.to_dict(), axis=1).tolist()
    rounded = df_sales_with_labels_rounded.apply(lambda row: row.to_dict(), axis=1).tolist()
    rounded_with_names = df_sales_with_labels_rounded_name.apply(lambda row: row.to_dict(), axis=1).tolist()

    return full_precision, rounded, rounded_with_names

@blueprint.route('/cluster')
@login_required
def index():

    return render_template('cluster/index.html')


@blueprint.route('/cluster/process', methods=['POST'])
@login_required
def process_cluster():
    try:
        iterations = int(request.form.get('cluster_count'))
        variables = request.form.getlist('variables')

        # Enforce required variables
        required_vars = {'perfume_id', 'age', 'gender', 'profession_id'}
        missing_vars = required_vars - set(variables)
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        if not variables:
            raise ValueError("No variables selected")
            
        if iterations < 1 or iterations > 100:
            raise ValueError("Iterations must be between 1-100")
        
        results, results_rounded, results_rounded_name = perform_kmeans(iterations, variables)
        return render_template('cluster/table.html', results=results, results_rounded=results_rounded, results_rounded_name=results_rounded_name)
            
    except Exception as e:
        current_app.logger.error(f"Cluster error: {str(e)}")
        return render_template('home/page-500.html'), 500


@blueprint.route('/cluster/results')
@login_required
def cluster_results():
    # Retrieve clusters from the database
    clusters = Cluster.query.all()
    cluster_data = []
    
    # Build cluster data with perfume composition per cluster
    for cluster in clusters:
        # Get all Result entries for this cluster
        result_entries = Result.query.filter_by(cluster_id=cluster.id).all()
        sales_ids = [r.sales_id for r in result_entries]
        # Query sales that belong to these results
        sales = Sale.query.filter(Sale.id.in_(sales_ids)).all()
        perfume_counts = {}
        total = 0
        for sale in sales:
            # Assuming a relationship exists: sale.perfume returns the Perfume object.
            # Otherwise, use: perfume = Perfume.query.get(sale.perfume_id)
            perfume = sale.perfume  
            perfume_name = perfume.name
            perfume_names = list(perfume_counts.keys())
            if perfume_name not in perfume_names:
                perfume_names.append(perfume_name)
            
            perfume_counts[perfume_name] = perfume_counts.get(perfume_name, 0) + 1
            total += 1
        cluster_data.append({
            'id': cluster.id,
            'label': cluster.label,
            'total': total,
            'perfume_counts': perfume_counts
        })
    return render_template('cluster/results.html', cluster_data=cluster_data, perfume_names=perfume_names)

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
