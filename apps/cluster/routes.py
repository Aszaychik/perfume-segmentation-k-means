# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.cluster import blueprint
from apps.cluster.models import Cluster, Result
from apps import db
from apps.sale.models import Sale
from flask import render_template, request, redirect, current_app
from flask_login import login_required
from jinja2 import TemplateNotFound
import os
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sqlalchemy import create_engine

def perform_kmeans(iterations, variables, centroid_ids=[5,10,15,20,25]):
    # Fetch sales data based on selected variables
    sales_data = Sale.query.with_entities(Sale.id, *[getattr(Sale, var) for var in variables]).all()
    
    if not sales_data:
        raise ValueError("No sales data available for clustering.")
    
    # Convert sales data to a DataFrame
    import pandas as pd
    df_sales = pd.DataFrame(sales_data, columns=['id'] + variables)

    # Extract initial centroids from the specified sales IDs
    initial_centroids = df_sales[df_sales['id'].isin(centroid_ids)][variables].values

    if len(initial_centroids) != len(centroid_ids):
        raise ValueError("Some centroid IDs were not found in sales data.")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=len(centroid_ids), init=initial_centroids, n_init=1, max_iter=iterations)
    df_sales['cluster'] = kmeans.fit_predict(df_sales[variables])

    # Store clusters in the database
    db.session.query(Result).delete()   # Clear previous result data
    db.session.query(Cluster).delete()  # Clear previous cluster data
    db.session.commit()

    # Save cluster data with all characteristics
    clusters = []
    for cluster_id in range(len(centroid_ids)):
        cluster_members = df_sales[df_sales['cluster'] == cluster_id]
        
        if cluster_members.empty:
            continue  # Skip empty clusters

        # Calculate cluster characteristics
        age_mean = cluster_members['age'].mean()
        perfume_mode = cluster_members['perfume_id'].mode()[0]
        gender_mode = cluster_members['gender'].mode()[0]
        profession_mode = cluster_members['profession_id'].mode()[0]

        new_cluster = Cluster(
            label=f"Age {round(age_mean)}",
            age=float(age_mean),
            perfume_id=float(perfume_mode),
            gender=float(gender_mode),
            profession_id=float(profession_mode)
        )
        
        db.session.add(new_cluster)
        clusters.append(new_cluster)
    
    db.session.commit()

    # Save result data
    for index, row in df_sales.iterrows():
        sales_id = row['id']
        cluster_id = row['cluster']
        new_result = Result(cluster_id=clusters[cluster_id].id, sales_id=sales_id)
        db.session.add(new_result)

    db.session.commit()
    
    return True

def kmeans_result_table():

    database_config = {
        'engine': os.getenv('DB_ENGINE'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'username': os.getenv('DB_USERNAME'),
        'pass': os.getenv('DB_PASS'),
        'database': os.getenv('DB_NAME')
    }

    connection_str = f"{database_config['engine']}://{database_config['username']}:{database_config['pass']}@{database_config['host']}:{database_config['port']}/{database_config['database']}"
    engine = create_engine(connection_str)
    
    # Convert sales data to a DataFrame
    query = """
            SELECT
                s.id,
                s.age,
                s.gender,
                s.profession_id,
                s.perfume_id,
                p.name AS perfume_name,
                pr.name AS profession_name
            FROM sales AS s
            LEFT JOIN perfumes AS p ON s.perfume_id = p.id
            LEFT JOIN professions AS pr ON s.profession_id = pr.id
            """
    df_sales = pd.read_sql(query, engine)

    # Select features from the DataFrame
    X = df_sales[['age', 'gender', 'profession_id', 'perfume_id']].copy()

    # Specify the sales IDs to use as initial centroids
    initial_centroid_ids = [5, 10, 15, 20, 25]

    # Extract the rows with these sales IDs to obtain the initial centroids
    # Make sure that the 'id' column corresponds to your sales id
    initial_centroids = df_sales[df_sales['id'].isin(initial_centroid_ids)][['age', 'gender', 'profession_id', 'perfume_id']].values

    # Initialize KMeans with the predefined centroids
    k = 5  # Total number of clusters
    kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1)

    # Train the KMeans model on the selected features
    kmeans.fit(X)

    # Retrieve the cluster labels and add them to the DataFrame
    labels = kmeans.labels_
    df_sales['cluster'] = labels

    # Display the first few cluster assignments
    df_sales['cluster'].head()

    # Get cluster labels
    df_sales_with_labels = df_sales.groupby('cluster').agg({
        'age': 'mean',
        'perfume_id': 'mean',
        'gender': 'mean',
        'profession_id': 'mean',
    })
    
    # Get luster labels rounded to the nearest integer
    df_sales_with_labels_rounded = df_sales_with_labels.round(0).astype(int)

    df_sales_with_labels_rounded_name = df_sales_with_labels_rounded.copy()

    df_sales_with_labels_rounded_name['perfume_name'] = df_sales.groupby('cluster')['perfume_name'].agg(lambda x: x.mode()[0])
    df_sales_with_labels_rounded_name['profession_name'] = df_sales.groupby('cluster')['profession_name'].agg(lambda x: x.mode()[0])

    return df_sales_with_labels.reset_index().apply(lambda row: row.to_dict(), axis=1).tolist(), df_sales_with_labels_rounded.reset_index().apply(lambda row: row.to_dict(), axis=1).tolist(), df_sales_with_labels_rounded_name.reset_index().apply(lambda row: row.to_dict(), axis=1).tolist()

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
        
        success = perform_kmeans(iterations, variables)
        
        if success:
            return redirect('/cluster/results')
        else:
            return render_template('home/page-500.html'), 500
            
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

@blueprint.route('/cluster/table')
@login_required
def cluster_table():
    try:
        results, results_rounded, results_rounded_name = kmeans_result_table()
        return render_template('cluster/table.html', results=results, results_rounded=results_rounded, results_rounded_name=results_rounded_name)
    except Exception as e:
        current_app.logger.error(f"Error generating cluster table: {str(e)}")
        return render_template('home/page-500.html'), 500

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
