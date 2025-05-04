# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re
import json
from datetime import datetime, timezone, timedelta  # Add timedelta to imports
# Add these imports at the top
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from dotenv import load_dotenv
# Load environment variables
load_dotenv()


app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'your-very-secret-key'),
    SESSION_COOKIE_NAME='rentease_tenant_session',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),  # Now this will work
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

# This config is for PRODUCTION
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://vanlam2024:2025StartupCenter@vanlam2024.mysql.pythonanywhere-services.com/vanlam2024$renteaseai'
# Google Gemini AI Setup
#os.environ['GOOGLE_API_KEY'] = "GOOGLE_API_KEY"

# This config is for DEV
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:121Deepmind@localhost/renteaseai'
# Google Gemini AI Setup
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

@app.template_filter('number_format')
def number_format(value, decimals=2):
    """Format numbers with commas and decimal places"""
    if value is None:
        return "Not specified"
    try:
        return "${:,.{}f}".format(float(value), decimals)
    except (ValueError, TypeError):
        return str(value)
    
from functools import wraps

def tenant_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not hasattr(current_user, 'tenant_id'):
            logout_user()
            return redirect(url_for('tenant_login'))
        return f(*args, **kwargs)
    return decorated_function

def landlord_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not hasattr(current_user, 'landlord_id'):
            logout_user()
            return redirect(url_for('landlord_login'))
        return f(*args, **kwargs)
    return decorated_function

app.jinja_env.filters['number_format'] = number_format

import markdown
from markupsafe import Markup

@app.template_filter('markdown_to_html')
def markdown_to_html(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    # Convert markdown to HTML
    html = markdown.markdown(text)
    # Mark as safe to render HTML
    return Markup(html)

app.jinja_env.filters['markdown_to_html'] = markdown_to_html

db = SQLAlchemy(app)
ai_chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@app.route('/session-info')
def session_info():
    return jsonify({
        'session': dict(session),
        'user_authenticated': current_user.is_authenticated,
        'user_id': current_user.get_id() if current_user.is_authenticated else None
    })

@app.before_request
def check_session():
    # Only run this in development for debugging
    if app.debug:
        print("\n--- Before Request ---")
        print(f"Path: {request.path}")
        print(f"Session: {dict(session)}")
        print(f"User authenticated: {current_user.is_authenticated}")
        if current_user.is_authenticated:
            print(f"User ID: {current_user.get_id()}")
            print(f"User type: {'tenant' if hasattr(current_user, 'tenant_id') else 'landlord'}")
        
# After db = SQLAlchemy(app), add:
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'tenant_login'  # Make sure this points to tenant login
login_manager.session_protection = "strong"  # Add this for better session security

#Add this user loader function
@login_manager.user_loader
def load_user(user_id):
    # Try loading as landlord first
    user = db.session.get(Landlord, int(user_id))
    if user is None:
        # If not landlord, try loading as tenant
        user = db.session.get(Tenant, int(user_id))
    return user

# @login_manager.user_loader
# def load_user(user_id):
#     try:
#         # First try loading as Tenant
#         tenant = Tenant.query.get(int(user_id))
#         if tenant:
#             return tenant
        
#         # Then try loading as Landlord
#         landlord = Landlord.query.get(int(user_id))
#         if landlord:
#             return landlord
        
#         return None
#     except:
#         return None

@app.context_processor
def utility_processor():
    def is_tenant(user):
        return user.is_authenticated and hasattr(user, 'tenant_id')
    return dict(is_tenant=is_tenant)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    # Get some stats to showcase
    total_properties = Property.query.count()
    avg_discount = db.session.query(db.func.avg(Property.discount_value)).scalar() or 0
    avg_vacancy_saved = 28  # Example - could calculate from your data

    return render_template('about.html',
                         total_properties=total_properties,
                         avg_discount=round(avg_discount, 1),
                         avg_vacancy_saved=avg_vacancy_saved)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Add this new route before the search route
@app.route('/sample_properties')
def sample_properties():
    # Get 6 random sample properties
    sample_props = Property.query.order_by(db.func.random()).limit(6).all()
    return jsonify({
        'properties': [p.to_dict() for p in sample_props]
    })

@app.route('/property/add', methods=['GET'])
@login_required
def add_property_form():
    return render_template('add-property.html')

@app.route('/property/create', methods=['POST'])
@login_required
def create_property():
    try:
        new_property = Property(
            landlord_id=current_user.landlord_id,
            address=request.form.get('address'),
            location=request.form.get('location'),
            description=request.form.get('description'),
            price=float(request.form.get('price')),
            bedrooms=int(request.form.get('bedrooms')),
            bathrooms=float(request.form.get('bathrooms')),
            sqft=int(request.form.get('sqft')),
            amenities=request.form.get('amenities'),
            available_date=datetime.strptime(request.form.get('available_date'), '%Y-%m-%d'),
            discount_value=float(request.form.get('discount_value', 0)))
        
        db.session.add(new_property)
        db.session.commit()
        
        # Return JSON for AJAX or redirect for form submission
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': 'Property added successfully'})
        else:
            return redirect(url_for('landlord_dashboard'))  # This goes to /landlord/dashboard
            
    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': str(e)}), 400
        else:
            flash('Error adding property: ' + str(e), 'error')
            return redirect(url_for('add_property_form'))

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/tenant')
def tenant():
    if current_user.is_authenticated and hasattr(current_user, 'tenant_id'):
        return redirect(url_for('tenant_dashboard'))
    return redirect(url_for('tenant_login'))

@app.route('/search', methods=['POST'])
def search():
    user_query = request.form.get('description', '').lower()
    location = request.form.get('location')
    price = request.form.get('price')
    date = request.form.get('date')

    # Extract requirements from natural language query
    max_price = extract_number(user_query, ['budget', 'max', '\$'])
    bedrooms = extract_number(user_query, ['bed', 'bedroom', 'br'])
    bathrooms = extract_number(user_query, ['bath', 'bathroom', 'ba'])

    # Use form inputs if not found in natural language
    bedrooms = bedrooms or request.form.get('bedrooms')
    max_price = max_price or price
    
    # Default understanding response
    understanding = {
        "explicit_requirements": [],
        "implicit_preferences": [],
        "clarification_questions": []
    }

    # Enhanced AI understanding prompt
    understanding_prompt = f"""Analyze this rental query: "{user_query}"
    Extract requirements and respond ONLY with JSON:
    {{
        "bedrooms": {bedrooms or 'null'},
        "bathrooms": {bathrooms or 'null'},
        "max_price": {max_price or 'null'},
        "must_have": ["list of absolute requirements"],
        "preferences": ["list of preferred features"],
        "neighborhood_qualities": ["list of desired neighborhood features"]
    }}"""

    try:
        ai_response = ai_chat.invoke(understanding_prompt).content
        ai_data = json.loads(ai_response)
        
        # Apply strict filters from AI analysis
        bedrooms = bedrooms or ai_data.get('bedrooms')
        bathrooms = bathrooms or ai_data.get('bathrooms')
        max_price = max_price or ai_data.get('max_price')
        
        understanding.update(ai_data)
    except Exception as e:
        print(f"AI analysis error: {e}")

    # Property Query with strict filtering
    query = Property.query
    
    # STRICT REQUIREMENTS
    if bedrooms:
        query = query.filter(Property.bedrooms >= int(bedrooms))  # At least X bedrooms
    if max_price:
        query = query.filter(Property.price <= float(max_price))
    if location:
        query = query.filter(Property.location.ilike(f'%{location}%'))
    
    # PREFERENCES (boost these in ranking)
    if 'safe' in understanding.get('neighborhood_qualities', []):
        query = query.filter(Property.description.ilike('%safe%') | 
                       Property.location.ilike('%safe%'))
    
    if 'school' in understanding.get('neighborhood_qualities', []):
        query = query.filter(Property.description.ilike('%school%') | 
                       Property.amenities.ilike('%school%'))
    
    if 'family' in user_query.lower():
        query = query.filter(Property.description.ilike('%family%') |
                       Property.amenities.ilike('%family%'))

    properties = query.order_by(Property.price.asc()).all()  # Sort by price

    # Generate summary with strict requirements highlighted
    summary_prompt = f"""Create a property search summary highlighting:

=== Strict Requirements ===
- Bedrooms: {bedrooms or 'Not specified'}
- Max Price: ${max_price if max_price else 'Not specified'}
- Location: {location or 'Any'}

=== Matching Properties ===
Found {len(properties)} properties:

{format_property_list(properties, bedrooms, max_price)}

=== Analysis ===
1. Highlight properties that match ALL strict requirements
2. Note any missing requirements
3. Suggest alternatives if few matches exist"""

    try:
        ai_summary = ai_chat.invoke(summary_prompt).content
    except Exception as e:
        print(f"AI summary error: {e}")
        ai_summary = f"Found {len(properties)} properties matching your criteria."

    return jsonify({
        'ai_summary': ai_summary,
        'properties': [p.to_dict() for p in properties],
        'clarification_questions': understanding.get('clarification_questions', [])
    })

# Then add this helper function:
def format_property_list(properties, target_beds=None, target_baths=None):
    """
    Formats a list of properties for display in the AI summary
    Args:
        properties: List of Property objects
        target_beds: Desired bedroom count (None for any)
        target_baths: Desired bathroom count (None for any)
    Returns:
        Formatted string with property details
    """
    if not properties:
        return "No properties found matching your criteria."

    property_lines = []

    for prop in properties:
        # Check if this property matches all specified criteria
        is_exact_match = True
        if target_beds is not None and prop.bedrooms != target_beds:
            is_exact_match = False
        if target_baths is not None and prop.bathrooms != target_baths:
            is_exact_match = False

        # Format the price nicely
        price_str = f"${prop.price:,.0f}" if prop.price else "Price not available"

        # Create the property line
        prop_str = f"{prop.address}: {prop.bedrooms} BR, {prop.bathrooms} BA, {price_str}"
        if is_exact_match:
            prop_str = f"**{prop_str} - EXACT MATCH**"

        property_lines.append(f"- {prop_str}")

    # Group exact matches first
    exact_matches = [line for line in property_lines if "EXACT MATCH" in line]
    other_properties = [line for line in property_lines if "EXACT MATCH" not in line]

    # Combine with section headers
    formatted = []
    if exact_matches:
        formatted.append("=== Exact Matches ===")
        formatted.extend(exact_matches)
    if other_properties:
        if exact_matches:
            formatted.append("")  # Empty line between sections
        formatted.append("=== Other Properties ===")
        formatted.extend(other_properties)

    return "\n".join(formatted)

def extract_number(query, keywords):
    """Extracts numeric values before keywords"""
    pattern = r'(\d+)\s*(' + '|'.join(keywords) + ')'
    match = re.search(pattern, query)
    return int(match.group(1)) if match else None

def find_matching_tenants(search_criteria):
    """Helper function to find tenants matching property criteria"""
    query = Tenant.query
    
    property_type = search_criteria.get('property_type', 'Any')
    rent_amount = search_criteria.get('rent_amount', 0)
    lease_term = search_criteria.get('lease_term', 'Any')
    tenant_requirements = search_criteria.get('tenant_requirements', '')
    discount_offer = search_criteria.get('discount_offer', 'None')
    
    if property_type and property_type != 'Any':
        query = query.filter(Tenant.preferred_type.ilike(f'%{property_type}%'))
    
    if rent_amount and rent_amount > 0:
        query = query.filter(Tenant.max_budget >= rent_amount)
    
    if lease_term and lease_term != 'Any':
        if lease_term == 'Short-term':
            query = query.filter(Tenant.discount_type.ilike('%short%'))
        elif lease_term == 'Long-term':
            query = query.filter(Tenant.discount_type.ilike('%long%'))
    
    if tenant_requirements:
        requirements = tenant_requirements.lower()
        if 'pet' in requirements:
            query = query.filter(Tenant.amenities_required.ilike('%pet%'))
        if 'parking' in requirements:
            query = query.filter(Tenant.amenities_required.ilike('%parking%'))
        if 'laundry' in requirements:
            query = query.filter(Tenant.amenities_required.ilike('%laundry%'))
    
    if discount_offer and discount_offer != 'None':
        query = query.filter(Tenant.discount_type.ilike(f'%{discount_offer}%'))
    
    matched_tenants = query.all()
    
    # Generate AI summary
    summary_prompt = f"""Analyze these tenant matches for a property with these features:
    - Type: {property_type or 'Any'}
    - Rent: ${rent_amount:,.2f}
    - Amenities: {tenant_requirements or 'None'}
    
    Found {len(matched_tenants)} matching tenants. Provide recommendations."""
    
    try:
        ai_summary = ai_chat.invoke(summary_prompt).content
    except Exception as e:
        ai_summary = f"Found {len(matched_tenants)} tenants matching this property."
    
    return {
        'tenants': [t.to_dict() for t in matched_tenants],
        'ai_summary': ai_summary
    }

@app.route('/landlord/find-tenants')
@login_required
def find_tenants_for_all_properties():
    # Get all properties for the current landlord
    properties = Property.query.filter_by(landlord_id=current_user.landlord_id).all()
    
    if not properties:
        flash("You don't have any properties listed yet.", "warning")
        return redirect(url_for('add_property_form'))
    
    property_tenant_data = []
    for property in properties:
        # Create search criteria based on property features
        search_criteria = {
            'property_type': extract_property_type(property.description),
            'rent_amount': property.price,
            'lease_term': 'Any',
            'tenant_requirements': property.amenities or '',
            'discount_offer': 'None'
        }
        
        # Find matching tenants
        matched_tenants = find_matching_tenants(search_criteria)
        
        property_tenant_data.append({
            'property': property.to_dict(),
            'tenants': matched_tenants['tenants'],
            'ai_summary': matched_tenants['ai_summary']
        })
    
    return render_template('tenant-results.html', 
                         properties_with_tenants=property_tenant_data,
                         view='all_properties')

def extract_property_type(description):
    """Extract property type from description"""
    if not description:
        return 'Any'
    
    desc_lower = description.lower()
    if 'apartment' in desc_lower:
        return 'Apartment'
    elif 'house' in desc_lower:
        return 'House'
    elif 'condo' in desc_lower:
        return 'Condo'
    elif 'townhouse' in desc_lower:
        return 'Townhouse'
    return 'Any'
    
@app.template_filter('markdown_to_html')
def markdown_to_html(markdown_text):
    # Simple markdown to HTML conversion
    if not markdown_text:
        return ""
    
    html = markdown_text.replace('**', '<strong>').replace('**', '</strong>')
    html = html.replace('*', '<em>').replace('*', '</em>')
    html = html.replace('\n', '<br>')
    return html

@app.route('/tenant/results', methods=['GET', 'POST'])
@login_required
def tenant_results():
    if request.method == 'POST':
        # Process the form data
        property_type = request.form.get('property_type', 'Any')
        rent_amount = float(request.form.get('rent_amount', 0)) if request.form.get('rent_amount') else 0
        lease_term = request.form.get('lease_term', 'Any')
        tenant_requirements = request.form.get('tenant_requirements', '')
        discount_offer = request.form.get('discount_offer', 'None')
        
        # Query tenants based on criteria
        query = Tenant.query
        
        if property_type and property_type != 'Any':
            query = query.filter(Tenant.preferred_type.ilike(f'%{property_type}%'))
        
        if rent_amount and rent_amount > 0:
            query = query.filter(Tenant.max_budget >= rent_amount)
        
        if lease_term and lease_term != 'Any':
            if lease_term == 'Short-term':
                query = query.filter(Tenant.discount_type.ilike('%short%'))
            elif lease_term == 'Long-term':
                query = query.filter(Tenant.discount_type.ilike('%long%'))
        
        if tenant_requirements:
            requirements = tenant_requirements.lower()
            if 'pet' in requirements:
                query = query.filter(Tenant.amenities_required.ilike('%pet%'))
            if 'parking' in requirements:
                query = query.filter(Tenant.amenities_required.ilike('%parking%'))
            if 'laundry' in requirements:
                query = query.filter(Tenant.amenities_required.ilike('%laundry%'))
        
        if discount_offer and discount_offer != 'None':
            query = query.filter(Tenant.discount_type.ilike(f'%{discount_offer}%'))
        
        matched_tenants = query.all()
        
        # Generate AI summary
        summary_prompt = f"""Analyze these tenant matches and create a summary:
        
        Landlord Requirements:
        - Property Type: {property_type or 'Any'}
        - Rent Amount: ${rent_amount:,.2f}
        - Lease Term: {lease_term or 'Any'}
        - Requirements: {tenant_requirements or 'None'}
        - Discount Offer: {discount_offer or 'None'}
        
        Found {len(matched_tenants)} matching tenants.
        
        Provide recommendations on which tenants might be the best fits."""
        
        try:
            ai_summary = ai_chat.invoke(summary_prompt).content
        except Exception as e:
            ai_summary = f"Found {len(matched_tenants)} tenants matching your criteria."
        
        return render_template('tenant-results.html', 
                            ai_summary=ai_summary,
                            tenants=matched_tenants,
                            search_criteria={
                                'property_type': property_type,
                                'rent_amount': rent_amount,
                                'lease_term': lease_term,
                                'tenant_requirements': tenant_requirements,
                                'discount_offer': discount_offer
                            })
    
    # If GET request, redirect back to search page
    return redirect(url_for('find_tenants'))

def format_tenant_list(tenants):
    """Formats tenant list for AI summary"""
    if not tenants:
        return "No matching tenants found."
    
    tenant_lines = []
    for tenant in tenants:
        # Handle None values in max_budget
        budget_str = f"${tenant.max_budget:,.2f}" if tenant.max_budget is not None else "Not specified"
        
        tenant_lines.append(
            f"- {tenant.first_name} {tenant.last_name}: "
            f"Budget {budget_str}, "
            f"Prefers {tenant.preferred_type or 'any'} property, "
            f"Needs {tenant.amenities_required or 'no specific amenities'}"
        )
    
    return "\n".join(tenant_lines)

class Property(db.Model):
    __tablename__ = 'Properties'

    property_id = db.Column(db.Integer, primary_key=True)  # Make sure this has primary_key=True
    landlord_id = db.Column(db.Integer, db.ForeignKey('Landlords.landlord_id'), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(50))
    description = db.Column(db.Text)
    price = db.Column(db.Float)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Float)
    sqft = db.Column(db.Integer)
    available_date = db.Column(db.Date)
    image_url = db.Column(db.String(255))
    amenities = db.Column(db.Text)
    early_bird_discount = db.Column(db.Float)
    long_term_discount = db.Column(db.Float)
    referral_discount = db.Column(db.Float)
    discount_type = db.Column(db.String(255))
    discount_value = db.Column(db.Float)
    
    # Relationship to Landlord
    landlord = db.relationship('Landlord', backref='properties')
    
    # Relationship to PropertyReview - updated to match new backref name
    reviews = db.relationship('PropertyReview', backref='review_property', lazy='dynamic')
    
    def to_dict(self):
        # First check for specific discount type/value
        if self.discount_type and self.discount_value:
            if "early" in self.discount_type.lower() or "percentage" in self.discount_type.lower():
                discounted_price = self.price * (1 - self.discount_value / 100)
            elif "fixed" in self.discount_type.lower():
                discounted_price = self.price - self.discount_value
            else:
                discounted_price = None
        else:
            # Fall back to the highest of the individual discounts
            max_discount = max(
                self.early_bird_discount or 0,
                self.long_term_discount or 0,
                self.referral_discount or 0
            )
            discounted_price = self.price * (1 - max_discount / 100) if max_discount > 0 else None

        # Calculate average rating and get reviews
        reviews = [r.to_dict() for r in self.reviews]
        avg_rating = db.session.query(
            db.func.avg(PropertyReview.rating)
        ).filter_by(property_id=self.property_id).scalar() or 0

        return {
            "property_id": self.property_id,
            "landlord_id": self.landlord_id,
            "address": self.address,
            "location": self.location,
            "description": self.description,
            "price": self.price,
            "discounted_price": round(discounted_price, 2) if discounted_price else None,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "sqft": self.sqft,
            "available_date": self.available_date.isoformat() if self.available_date else None,
            "image_url": self.image_url,
            "amenities": self.amenities,
            "discount_type": self.discount_type,
            "discount_value": self.discount_value,
            "discount_available": (discounted_price is not None),
            # New review-related fields
            "reviews": reviews,
            "average_rating": round(float(avg_rating), 1),
            "review_count": len(reviews)
        }

# Add UserMixin to your Landlord model (you'll need to create this class)
class Landlord(db.Model, UserMixin):
    __tablename__ = 'Landlords'
    
    landlord_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    phone_number = db.Column(db.String(20))
    bio = db.Column(db.Text)
    location = db.Column(db.String(255))
    password_hash = db.Column(db.String(255))  # Increased from 128 to 255
    
    def get_id(self):
        return str(self.landlord_id)
    
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

# Add these new routes before your existing landlord route
@app.route('/landlord/login', methods=['GET', 'POST'])
def landlord_login():
    if current_user.is_authenticated and hasattr(current_user, 'landlord_id'):
        return redirect(url_for('landlord_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        landlord = Landlord.query.filter_by(email=email).first()
        
        if not landlord or not landlord.verify_password(password):
            flash('Invalid email or password', 'error')
            return render_template('landlord-login.html', error='Invalid credentials')
        
        login_user(landlord, remember=remember)
        next_page = request.args.get('next') or url_for('landlord_dashboard')
        return redirect(next_page)
    
    return render_template('landlord-login.html')


@app.route('/landlord/signup', methods=['GET', 'POST'])
def landlord_signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if email already exists
        if Landlord.query.filter_by(email=email).first():
            return render_template('landlord-signup.html', error='Email already registered')
        
        # Create new landlord
        new_landlord = Landlord(
            email=email,
            first_name=request.form.get('first_name'),
            last_name=request.form.get('last_name'),
            phone_number=request.form.get('phone_number')
        )
        new_landlord.password = password
        
        db.session.add(new_landlord)
        db.session.commit()
        
        login_user(new_landlord)
        return redirect(url_for('landlord_dashboard'))
    
    return render_template('landlord-signup.html')

@app.route('/landlord/logout')
@login_required
def landlord_logout():
    logout_user()
    return redirect(url_for('home'))

# Modify your existing landlord route
@app.route('/landlord')
@login_required
def landlord():
    return redirect(url_for('landlord_dashboard'))

@app.route('/landlord/dashboard')
@login_required
@landlord_required
def landlord_dashboard():
    # Check if current user is a landlord
    if not hasattr(current_user, 'landlord_id'):
        logout_user()  # Log out the tenant user
        return redirect(url_for('landlord_login'))
    
    # Rest of your existing code...
    
    # Get properties for the current landlord
    properties = Property.query.filter_by(landlord_id=current_user.landlord_id).all()
    
    # Calculate dashboard statistics
    total_properties = len(properties)
    
    # Calculate occupied properties
    occupied = len([p for p in properties if p.available_date > datetime.now(timezone.utc).date()])
    
    # Calculate vacancy rate (percentage)
    vacancy_rate = round((1 - occupied/total_properties)*100, 1) if total_properties else 0
    
    # Calculate average rent
    avg_rent = db.session.query(db.func.avg(Property.price)).filter_by(
        landlord_id=current_user.landlord_id
    ).scalar() or 0
    
    # Count pending applications (you'll need to implement this based on your data model)
    pending_applications = 0  # Placeholder - replace with actual query
    
    return render_template('landlord.html',
                         properties=properties,
                         today=datetime.now(timezone.utc).date(),
                         vacancy_rate=vacancy_rate,
                         avg_rent=avg_rent,
                         pending_applications=pending_applications,
                         view='dashboard')

def get_properties_for_landlord(landlord_id):
    """Helper function to get properties for a specific landlord"""
    return Property.query.filter_by(landlord_id=landlord_id).all()

@app.route('/landlord/properties')
@login_required
def landlord_properties():
    properties = get_properties_for_landlord(current_user.landlord_id)
    # Calculate stats for the properties view
    total_properties = len(properties)
    occupied = len([p for p in properties if p.available_date > datetime.now(timezone.utc).date()])
    vacancy_rate = round((1 - occupied/total_properties)*100, 1) if total_properties else 0
    
    return render_template('landlord.html',
                         properties=properties,
                         today=datetime.now(timezone.utc).date(),
                         vacancy_rate=vacancy_rate,
                         view='properties')  # Add a view parameter

class Tenant(db.Model, UserMixin):
    __tablename__ = 'Tenants'
    
    tenant_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    phone_number = db.Column(db.String(20))
    password_hash = db.Column(db.String(255))  # Add this line
    rental_application = db.Column(db.Text)
    rental_application_status = db.Column(db.String(50))
    location_preference = db.Column(db.String(255))
    max_budget = db.Column(db.Float, default=0)
    min_bedrooms = db.Column(db.Integer, default=0)
    min_bathrooms = db.Column(db.Float, default=0)
    preferred_type = db.Column(db.String(50), default='Any')
    discount_type = db.Column(db.String(50))
    discount_value = db.Column(db.Float)
    move_in_date = db.Column(db.Date)
    amenities_required = db.Column(db.String(255))
    
    def is_active(self):
        """Required by Flask-Login"""
        return True
    
    def is_authenticated(self):
        """Required by Flask-Login"""
        return True
    
    def is_anonymous(self):
        """Required by Flask-Login"""
        return False 
    
    def get_id(self):
        return str(self.tenant_id)
    
    def get_id(self):
        """Override to ensure string return"""
        return str(self.tenant_id)
    
    @property
    def is_active(self):
        """All users are active by default"""
        return True
    
    @property
    def is_authenticated(self):
        """Return True if authenticated"""
        return True
    
    @property
    def is_anonymous(self):
        """False for actual users"""
        return False
    
    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            "id": self.tenant_id,  # Changed from "tenant_id" to match what the frontend expects
            "first_name": self.first_name or "",  # Ensure we always return a string
            "last_name": self.last_name or "",    # Ensure we always return a string
            "email": self.email,
            "phone": self.phone_number,
            "application_status": self.rental_application_status,
            "location_preference": self.location_preference,
            "max_budget": self.max_budget,
            "preferred_type": self.preferred_type,
            "min_bedrooms": self.min_bedrooms,
            "min_bathrooms": self.min_bathrooms,
            "move_in_date": self.move_in_date.isoformat() if self.move_in_date else None,
            "amenities_required": self.amenities_required
    }

@app.route('/tenant/login', methods=['GET', 'POST'])
def tenant_login():
    if hasattr(current_user, 'tenant_id'):
        return redirect(url_for('tenant_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        # Find the tenant by email only
        tenant = Tenant.query.filter_by(email=email).first()
        
        if not tenant:
            flash('User not found', 'error')
            return render_template('tenant-login.html', error='Email not found')
        
        # Skip password verification and log in the user directly
        login_user(tenant, remember=True)
        next_page = request.args.get('next') or url_for('tenant_dashboard')
        return redirect(next_page)
    
    return render_template('tenant-login.html')

@app.route('/tenant/signup', methods=['GET', 'POST'])
def tenant_signup():
    if request.method == 'POST':
        try:
            new_tenant = Tenant(
                email=request.form.get('email'),
                first_name=request.form.get('first_name'),
                last_name=request.form.get('last_name'),
                phone_number=request.form.get('phone_number'),
                max_budget=float(request.form.get('max_budget', 0)),
                preferred_type=request.form.get('preferred_type', 'Any'),
                min_bedrooms=int(request.form.get('min_bedrooms', 0)),
                min_bathrooms=float(request.form.get('min_bathrooms', 0))
            )
            new_tenant.password = request.form.get('password')
            
            db.session.add(new_tenant)
            db.session.commit()
            
            login_user(new_tenant)
            return redirect(url_for('tenant_dashboard'))
            
        except ValueError as e:
            flash('Please enter valid numbers for budget and room requirements', 'error')
            return render_template('tenant-signup.html')
    
    return render_template('tenant-signup.html')

@app.route('/tenant/logout')
def tenant_logout():
    #logout_user()
    return redirect(url_for('home'))

@app.route('/tenant/dashboard')
def tenant_dashboard():
    if not hasattr(current_user, 'tenant_id'):
        #logout_user()
        return redirect(url_for('tenant_login'))
    
    # Safely handle None values in preferences
    tenant_preferences = {
        'name': f"{current_user.first_name or ''} {current_user.last_name or ''}".strip(),
        'email': current_user.email or 'Not provided',
        'phone': current_user.phone_number or 'Not provided',
        'budget': current_user.max_budget if current_user.max_budget is not None else 0,
        'preferred_type': current_user.preferred_type or 'Not specified',
        'bedrooms': current_user.min_bedrooms if current_user.min_bedrooms is not None else 'Any',
        'bathrooms': current_user.min_bathrooms if current_user.min_bathrooms is not None else 'Any'
    }
    
    return render_template('tenant-dashboard.html', preferences=tenant_preferences)

class PropertyReview(db.Model):
    __tablename__ = 'propertyreviews'

    review_id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey('Properties.property_id'))
    tenant_id = db.Column(db.Integer, db.ForeignKey('Tenants.tenant_id'))
    review_date = db.Column(db.Date, default=datetime.utcnow)
    rating = db.Column(db.Integer)
    review_text = db.Column(db.Text)

    # Relationships - change the backref name to be more specific
    # Fix relationship conflicts with overlaps parameter
    property_rel = db.relationship('Property', backref='property_reviews', overlaps="review_property,reviews") 
    tenant = db.relationship('Tenant', backref='tenant_reviews')

    def to_dict(self):
        return {
            "review_id": self.review_id,
            "property_id": self.property_id,
            "tenant_id": self.tenant_id,
            "tenant_name": f"{self.tenant.first_name} {self.tenant.last_name}" if self.tenant else "Anonymous",
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "rating": self.rating,
            "review_text": self.review_text
        }


@app.route('/property/<int:property_id>/reviews')
def get_property_reviews(property_id):
    reviews = PropertyReview.query.filter_by(property_id=property_id).order_by(PropertyReview.review_date.desc()).all()
    return jsonify([review.to_dict() for review in reviews])

@app.route('/property/<int:property_id>/review', methods=['POST'])
@login_required
def add_property_review(property_id):
    if not hasattr(current_user, 'tenant_id'):
        return jsonify({"error": "Only tenants can leave reviews"}), 403
    
    data = request.get_json()
    
    # Check if tenant already reviewed this property
    existing_review = PropertyReview.query.filter_by(
        property_id=property_id,
        tenant_id=current_user.tenant_id
    ).first()
    
    if existing_review:
        return jsonify({"error": "You've already reviewed this property"}), 400
    
    try:
        new_review = PropertyReview(
            property_id=property_id,
            tenant_id=current_user.tenant_id,
            rating=data.get('rating'),
            review_text=data.get('review_text'),
            review_date=datetime.now(timezone.utc)
        )
        
        db.session.add(new_review)
        db.session.commit()
        
        # Update property's average rating (optional)
        property = Property.query.get(property_id)
        if property:
            avg_rating = db.session.query(
                db.func.avg(PropertyReview.rating)
                .filter_by(property_id=property_id)
                .scalar())
            # You could store this avg_rating in the Property model if you add a column
        
        return jsonify(new_review.to_dict()), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

@app.route('/review/<int:review_id>', methods=['DELETE'])
@login_required
def delete_review(review_id):
    review = PropertyReview.query.get_or_404(review_id)
    
    # Check if current user is the reviewer or an admin
    if not (hasattr(current_user, 'tenant_id') and current_user.tenant_id == review.tenant_id):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        db.session.delete(review)
        db.session.commit()
        return jsonify({"message": "Review deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400

@app.route('/tenant/preferences/edit', methods=['GET', 'POST'])
@login_required
def edit_tenant_preferences():
    if not hasattr(current_user, 'tenant_id'):
        return redirect(url_for('tenant_login'))
    
    if request.method == 'POST':
        try:
            current_user.max_budget = float(request.form.get('max_budget', 0))
            current_user.preferred_type = request.form.get('preferred_type')
            current_user.min_bedrooms = int(request.form.get('min_bedrooms', 0))
            current_user.min_bathrooms = float(request.form.get('min_bathrooms', 0))
            db.session.commit()
            return redirect(url_for('tenant_dashboard'))
        except Exception as e:
            return render_template('edit-preferences.html', error=str(e), preferences=current_user)
    
    return render_template('edit-preferences.html', preferences=current_user)

@app.route('/tenant/search')
@login_required
def tenant_search():
    if not hasattr(current_user, 'tenant_id'):
        return redirect(url_for('tenant_login'))
    
    # Get properties matching tenant's preferences
    query = Property.query
    
    if current_user.max_budget:
        query = query.filter(Property.price <= current_user.max_budget)
    if current_user.min_bedrooms:
        query = query.filter(Property.bedrooms >= current_user.min_bedrooms)
    if current_user.min_bathrooms:
        query = query.filter(Property.bathrooms >= current_user.min_bathrooms)
    if current_user.preferred_type:
        query = query.filter(Property.description.ilike(f'%{current_user.preferred_type}%'))
    
    properties = query.all()
    
    return render_template('tenant.html', properties=properties)


if __name__ == '__main__':
    app.run(debug=True)
