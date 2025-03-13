# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from datetime import datetime
from jinja2 import TemplateNotFound
from apps.sale.models import Sale
from apps.perfume.models import Perfume
from apps.cluster.models import Cluster, Result

@blueprint.route('/index')
@login_required
def index():
    sales = Sale.query.all()
    current_month = datetime.now().month

    total_sales = len(sales)
    monthly_sales = len([sale for sale in sales if sale.createdAt.month == current_month])

    most_sold_perfume_id = max(set([sale.perfume_id for sale in sales]), key=[sale.perfume_id for sale in sales].count)
    total_sold_most_sold_perfume = len([sale for sale in sales if sale.perfume_id == most_sold_perfume_id])
    most_sold_perfume = Perfume.query.get(most_sold_perfume_id)

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
        
    # Compute the highest cluster (with maximum total sales)
    highest_cluster = max(cluster_data, key=lambda x: x['total']) if cluster_data else None

    return render_template('home/index.html', segment='index', sales=sales, total_sales=total_sales, monthly_sales=monthly_sales, most_sold_perfume=most_sold_perfume, total_sold_most_sold_perfume=total_sold_most_sold_perfume, cluster_data=cluster_data, perfume_names=perfume_names, highest_cluster=highest_cluster)


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
