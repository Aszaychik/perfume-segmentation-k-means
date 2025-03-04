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
import math

def perform_kmeans(iterations, variables, centroid_ids=[5,10,15,20,25]):
    # 1. Get all sales data
    sales = Sale.query.order_by(Sale.id).all()
    if not sales:
        raise ValueError("No sales data available for clustering")
    
    # 2. Prepare data points using selected variables
    data_points = []
    for sale in sales:
        point = []
        if 'perfume_id' in variables:
            point.append(sale.perfume_id)
        if 'profession' in variables:
            point.append(sale.profession_id)
        if 'age' in variables:
            point.append(sale.age)
        if 'gender' in variables:
            point.append(sale.gender)
        data_points.append((sale.id, point))
    
    # 3. Validate and get initial centroids (fixed number based on predefined IDs)
    centroids = []
    valid_centroid_ids = [cid for cid in centroid_ids if Sale.query.get(cid)]
    
    if not valid_centroid_ids:
        raise ValueError("No valid centroid IDs found in the system")
    
    for cid in valid_centroid_ids:
        sale = Sale.query.get(cid)
        point = []
        if 'perfume_id' in variables:
            point.append(sale.perfume_id)
        if 'profession' in variables:
            point.append(sale.profession_id)
        if 'age' in variables:
            point.append(sale.age)
        if 'gender' in variables:
            point.append(sale.gender)
        centroids.append(point)
    
    k = len(centroids)  # Number of clusters determined by valid centroid IDs
    
    # 4. Custom K-Means implementation with iteration limit
    dimension = len(data_points[0][1]) if data_points else 0
    if dimension == 0:
        raise ValueError("No variables selected for clustering")
    
    for _ in range(iterations):
        # Assign points to clusters
        clusters = [[] for _ in range(k)]
        for pid, point in data_points:
            distances = [math.sqrt(sum((p-c)**2 for p,c in zip(point, centroid))) 
                        for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(pid)
        
        # Update centroids
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                sum_point = [0]*dimension
                for pid in clusters[i]:
                    point = next(p[1] for p in data_points if p[0] == pid)
                    for j in range(dimension):
                        sum_point[j] += point[j]
                mean_point = [v/len(clusters[i]) for v in sum_point]
                new_centroids.append(mean_point)
            else:
                new_centroids.append(centroids[i])
        
        centroids = new_centroids
    
    # 5. Save results to database
    try:
        Cluster.query.delete()
        Result.query.delete()
        
        for i in range(k):
            cluster = Cluster(
                label=f"Cluster {i+1}",
            )
            db.session.add(cluster)
            db.session.flush()
            
            for sale_id in clusters[i]:
                result = Result(
                    cluster_id=cluster.id,
                    sales_id=sale_id
                )
                db.session.add(result)
        
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error saving clusters: {str(e)}")
        return False

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
    clusters = Cluster.query.all()
    results = []
    for cluster in clusters:
        sales_count = Result.query.filter_by(cluster_id=cluster.id).count()
        results.append({
            'id': cluster.id,
            'label': cluster.label,
            'count': sales_count
        })
    return render_template('cluster/results.html', results=results)

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
