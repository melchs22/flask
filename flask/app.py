import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from flask_session import Session
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
import requests
import io
import base64
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize Supabase
def init_supabase():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url.startswith("https://"):
        url = f"https://{url}"
    return create_client(url, key)

# Check database tables
def check_db(supabase):
    required_tables = ["users", "kpis", "performance", "zoho_agent_data", "goals", "feedback", "notifications", "audio_assessments", "badges", "forum_posts"]
    critical_tables = ["users", "goals", "feedback", "performance"]
    missing_critical = []
    missing_non_critical = []
    
    for table in required_tables:
        try:
            supabase.table(table).select("count").limit(1).execute()
        except Exception:
            if table in critical_tables:
                missing_critical.append(table)
            else:
                missing_non_critical.append(table)
    
    return missing_critical, missing_non_critical

# Save KPIs
def save_kpis(supabase, kpis):
    try:
        for metric, threshold in kpis.items():
            response = supabase.table("kpis").select("*").eq("metric", metric).execute()
            if not response.data:
                supabase.table("kpis").insert({"metric": metric, "threshold": threshold}).execute()
            else:
                supabase.table("kpis").update({"threshold": threshold}).eq("metric", metric).execute()
        return True
    except Exception:
        return False

# Get KPIs
def get_kpis(supabase):
    try:
        response = supabase.table("kpis").select("*").execute()
        kpis = {}
        for row in response.data:
            metric = row["metric"]
            value = row["threshold"]
            kpis[metric] = int(float(value)) if metric == "call_volume" else float(value) if value is not None else 0.0
        return kpis
    except Exception:
        return {}

# Save performance data
def save_performance(supabase, agent_name, data):
    try:
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        performance_data = {
            "agent_name": agent_name,
            "attendance": data['attendance'],
            "quality_score": data['quality_score'],
            "product_knowledge": data['product_knowledge'],
            "contact_success_rate": data['contact_success_rate'],
            "onboarding": data['onboarding'],
            "reporting": data['reporting'],
            "talk_time": data['talk_time'],
            "resolution_rate": data['resolution_rate'],
            "aht": data['aht'],
            "csat": data['csat'],
            "call_volume": data['call_volume'],
            "date": date
        }
        supabase.table("performance").insert(performance_data).execute()
        kpis = get_kpis(supabase)
        for metric, value in performance_data.items():
            if metric in kpis and metric not in ["agent_name", "date"]:
                threshold = kpis[metric]
                badge_name = f"{metric.replace('_', ' ').title()} Star"
                if (metric == "aht" and value <= threshold * 0.9) or (metric != "aht" and value >= threshold * 1.1):
                    description = f"Achieved exceptional {metric.replace('_', ' ')} of {value:.1f}{' sec' if metric == 'aht' else '%'}"
                    award_badge(supabase, agent_name, badge_name, description, "System")
                if (metric == "aht" and value <= threshold * 0.9) or (metric != "aht" and value >= threshold * 1.1):
                    send_performance_alert(supabase, agent_name, metric, value, threshold, is_positive=True)
                elif (metric == "aht" and value > threshold * 1.1) or (metric != "aht" and value < threshold * 0.9):
                    send_performance_alert(supabase, agent_name, metric, value, threshold, is_positive=False)
        update_goal_status(supabase, agent_name)
        return True
    except Exception:
        return False

# Get performance data
def get_performance(supabase, agent_name=None):
    try:
        query = supabase.table("performance").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            numeric_cols = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
                           'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'call_volume' in df.columns:
                df['call_volume'] = pd.to_numeric(df['call_volume'], errors='coerce').fillna(0).astype(int)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Get Zoho agent data
def get_zoho_agent_data(supabase, agent_name=None):
    try:
        all_data = []
        chunk_size = 1000
        offset = 0
        while True:
            query = supabase.table("zoho_agent_data").select("*").range(offset, offset + chunk_size - 1)
            if agent_name:
                query = query.eq("ticket_owner", agent_name)
            response = query.execute()
            if not response.data:
                break
            all_data.extend(response.data)
            if len(response.data) < chunk_size:
                break
            offset += chunk_size
        if all_data:
            df = pd.DataFrame(all_data)
            if 'id' not in df.columns or 'ticket_owner' not in df.columns:
                return pd.DataFrame()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Set agent goal
def set_agent_goal(supabase, agent_name, metric, target_value, manager_name, is_manager=False):
    try:
        goal_data = {
            "agent_name": agent_name,
            "metric": metric,
            "target_value": target_value,
            "status": "Approved" if is_manager else "Awaiting Approval"
        }
        response = supabase.table("goals").select("*").eq("agent_name", agent_name).eq("metric", metric).execute()
        if response.data:
            supabase.table("goals").update(goal_data).eq("agent_name", agent_name).eq("metric", metric).execute()
        else:
            supabase.table("goals").insert(goal_data).execute()
        return True
    except Exception:
        return False

# Approve or reject goal
def approve_goal(supabase, goal_id, manager_name, approve=True):
    try:
        update_data = {
            "status": "Approved" if approve else "Rejected",
            "approved_at": datetime.now().isoformat()
        }
        supabase.table("goals").update(update_data).eq("id", goal_id).execute()
        if session.get("notifications_enabled", False):
            goal = supabase.table("goals").select("agent_name").eq("id", goal_id).execute()
            if goal.data:
                agent_name = goal.data[0]["agent_name"]
                agent = supabase.table("users").select("id").eq("name", agent_name).execute()
                if agent.data:
                    status = "approved" if approve else "rejected"
                    supabase.table("notifications").insert({
                        "user_id": agent.data[0]["id"],
                        "message": f"Your goal for {agent_name} was {status} by {manager_name}"
                    }).execute()
        return True
    except Exception:
        return False

# Update goal status
def update_goal_status(supabase, agent_name):
    try:
        goals = supabase.table("goals").select("*").eq("agent_name", agent_name).in_("status", ["Approved", "Pending"]).execute()
        if not goals.data:
            return
        perf = get_performance(supabase, agent_name)
        if perf.empty:
            return
        perf['date'] = pd.to_datetime(perf['date'], errors='coerce')
        if perf['date'].isna().all():
            return
        latest_perf = perf[perf['date'] == perf['date'].max()]
        if latest_perf.empty:
            return
        for goal in goals.data:
            metric = goal['metric']
            target = float(goal['target_value'])
            if metric in latest_perf.columns:
                value = float(latest_perf[metric].iloc[0])
                if (metric == "aht" and value <= target) or (metric != "aht" and value >= target):
                    status = "Completed"
                    badge_name = f"{metric.replace('_', ' ').title()} Master"
                    description = f"Achieved {metric} goal of {target:.1f}{' sec' if metric == 'aht' else '%'}"
                    award_badge(supabase, agent_name, badge_name, description, "System")
                else:
                    status = goal['status']
                supabase.table("goals").update({"status": status}).eq("id", goal['id']).execute()
    except Exception:
        pass

# Get feedback
def get_feedback(supabase, agent_name=None):
    try:
        query = supabase.table("feedback").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Respond to feedback
def respond_to_feedback(supabase, feedback_id, manager_response, manager_name):
    try:
        response_data = {
            "manager_response": manager_response,
            "response_timestamp": datetime.now().isoformat()
        }
        supabase.table("feedback").update(response_data).eq("id", feedback_id).execute()
        if session.get("notifications_enabled", False):
            feedback = supabase.table("feedback").select("agent_name").eq("id", feedback_id).execute()
            if feedback.data:
                agent_name = feedback.data[0]["agent_name"]
                agent = supabase.table("users").select("id").eq("name", agent_name).execute()
                if agent.data:
                    supabase.table("notifications").insert({
                        "user_id": agent.data[0]["id"],
                        "message": f"Manager responded to your feedback: {manager_response[:50]}..."
                    }).execute()
        return True
    except Exception:
        return False

# Get notifications
def get_notifications(supabase):
    if not session.get("notifications_enabled", False):
        return pd.DataFrame()
    try:
        user_response = supabase.table("users").select("id").eq("name", session.get('user')).execute()
        if not user_response.data:
            return pd.DataFrame()
        user_id = user_response.data[0]["id"]
        response = supabase.table("notifications").select("*").eq("user_id", user_id).eq("read", False).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Send performance alert
def send_performance_alert(supabase, agent_name, metric, value, threshold, is_positive=True):
    try:
        agent = supabase.table("users").select("id").eq("name", agent_name).execute()
        if agent.data:
            message = f"{'Great job' if is_positive else 'Attention'}: {metric.replace('_', ' ').title()} {'exceeded' if is_positive else 'below'} {threshold:.1f}{' sec' if metric == 'aht' else '%'} with {value:.1f}{' sec' if metric == 'aht' else '%'}"
            supabase.table("notifications").insert({
                "user_id": agent.data[0]["id"],
                "message": message
            }).execute()
        return True
    except Exception:
        return False

# Award badge
def award_badge(supabase, agent_name, badge_name, description, awarded_by):
    try:
        existing = supabase.table("badges").select("id").eq("agent_name", agent_name).eq("badge_name", badge_name).execute()
        if existing.data:
            return False
        supabase.table("badges").insert({
            "agent_name": agent_name,
            "badge_name": badge_name,
            "description": description,
            "awarded_by": awarded_by,
            "earned_at": datetime.now().isoformat()
        }).execute()
        if session.get("notifications_enabled", False):
            agent = supabase.table("users").select("id").eq("name", agent_name).execute()
            if agent.data:
                supabase.table("notifications").insert({
                    "user_id": agent.data[0]["id"],
                    "message": f"You earned the '{badge_name}' badge: {description}"
                }).execute()
        return True
    except Exception:
        return False

# Get leaderboard
def get_leaderboard(supabase):
    try:
        response = supabase.table("performance").select("agent_name").execute()
        if response.data:
            df_perf = pd.DataFrame(response.data)
            all_perf_response = supabase.table("performance").select("*").execute()
            if not all_perf_response.data:
                return pd.DataFrame()
            df_all = pd.DataFrame(all_perf_response.data)
            kpis = get_kpis(supabase)
            results = assess_performance(df_all, kpis)
            leaderboard_df = results.groupby("agent_name")["overall_score"].mean().reset_index()
            badges_response = supabase.table("badges").select("agent_name, id").execute()
            badges_df = pd.DataFrame(badges_response.data) if badges_response.data else pd.DataFrame(columns=["agent_name", "id"])
            badge_counts = badges_df.groupby("agent_name")["id"].nunique().reset_index(name="badges_earned")
            leaderboard_df = leaderboard_df.merge(badge_counts, on="agent_name", how="left").fillna({"badges_earned": 0})
            leaderboard_df["badges_earned"] = leaderboard_df["badges_earned"].astype(int)
            leaderboard_df = leaderboard_df.sort_values("overall_score", ascending=False)
            return leaderboard_df
    except Exception:
        return pd.DataFrame()

# Create forum post
def create_forum_post(supabase, user_name, message, category):
    try:
        supabase.table("forum_posts").insert({
            "user_name": user_name,
            "message": message,
            "category": category,
            "created_at": datetime.now().isoformat()
        }).execute()
        return True
    except Exception:
        return False

# Get forum posts
def get_forum_posts(supabase, category=None):
    try:
        query = supabase.table("forum_posts").select("user_name, message, category, created_at")
        if category:
            query = query.eq("category", category)
        response = query.order("created_at", desc=True).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            badge_counts = supabase.table("badges").select("agent_name, id").execute()
            badge_dict = {}
            if badge_counts.data:
                badges_df = pd.DataFrame(badge_counts.data)
                badge_dict = badges_df.groupby("agent_name")["id"].nunique().to_dict()
            df['badge_count'] = df['user_name'].map(badge_dict).fillna(0).astype(int)
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d')
            return df
    except Exception:
        return pd.DataFrame()

# Get AI coaching tips
def get_coaching_tips(supabase, agent_name):
    try:
        perf = get_performance(supabase, agent_name)
        if perf.empty:
            return []
        latest_perf = perf[perf['date'] == perf['date'].max()]
        kpis = get_kpis(supabase)
        tips = []
        api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not api_token:
            return []
        for metric in ['attendance', 'quality_score', 'csat', 'aht']:
            if metric in latest_perf.columns:
                value = float(latest_perf[metric].iloc[0])
                threshold = kpis.get(metric, 600 if metric == 'aht' else 50)
                if (metric == "aht" and value > threshold) or (metric != "aht" and value < threshold):
                    prompt = f"You are a call center coach. Provide a concise, actionable coaching tip for an agent whose {metric.replace('_', ' ')} is {value:.1f}{' sec' if metric == 'aht' else '%'}, below the target of {threshold:.1f}{' sec' if metric == 'aht' else '%'}."
                    headers = {"Authorization": f"Bearer {api_token}"}
                    response = requests.post(
                        "https://api-inference.huggingface.co/models/google/flan-t5-small",
                        headers=headers,
                        json={"inputs": prompt, "parameters": {"max_length": 50}}
                    )
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            tip = data[0]['generated_text'].strip() if isinstance(data, list) and data else "Focus on improving efficiency."
                        except (ValueError, KeyError):
                            tip = f"Focus on improving {metric.replace('_', ' ')}."
                    else:
                        tip = f"Focus on improving {metric.replace('_', ' ')}."
                    tips.append({"metric": metric, "tip": tip})
        return tips
    except Exception:
        return []

# Ask the AI coach
def ask_coach(supabase, agent_name, question):
    try:
        perf = get_performance(supabase, agent_name)
        context = ""
        if not perf.empty:
            latest_perf = perf[perf['date'] == perf['date'].max()]
            metrics = ['attendance', 'quality_score', 'csat', 'aht']
            context = "Agent's latest performance: " + ", ".join(
                f"{m.replace('_', ' ')}: {float(latest_perf[m].iloc[0]):.1f}{' sec' if m == 'aht' else '%'}"
                for m in metrics if m in latest_perf.columns
            ) + ". "
        api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not api_token:
            return "Please consult your manager."
        prompt = f"You are a call center coach. {context}Answer the agent's question concisely: {question}"
        headers = {"Authorization": f"Bearer {api_token}"}
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-small",
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_length": 100}}
        )
        if response.status_code == 200:
            try:
                data = response.json()
                answer = data[0]['generated_text'].strip() if isinstance(data, list) and data else "Please consult your manager."
            except (ValueError, KeyError):
                answer = "Please consult your manager."
        else:
            answer = "Please consult your manager."
        return answer
    except Exception:
        return "Please consult your manager."

# Plot interactive performance chart
def plot_performance_chart(supabase, agent_name=None, metrics=None):
    try:
        df = get_performance(supabase, agent_name)
        if df.empty:
            return None
        if metrics is None:
            metrics = ['attendance', 'quality_score', 'csat', 'resolution_rate']
        if agent_name:
            latest_df = df[df['date'] == df['date'].max()]
            values = [latest_df[m].mean() for m in metrics]
            fig = go.Figure(data=go.Scatterpolar(r=values, theta=[m.replace('_', ' ').title() for m in metrics], fill='toself'))
            fig.update_layout(title=f"Performance Profile for {agent_name}", polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        else:
            avg_df = df.groupby('agent_name')[metrics].mean().reset_index()
            fig = px.bar(avg_df, x='agent_name', y=metrics, barmode='group', title="Team Performance Comparison")
            fig.update_layout(yaxis_title="Value (%)", xaxis_title="Agent")
        return fig
    except Exception:
        return None

# Assess performance
def assess_performance(performance_df, kpis):
    if performance_df.empty:
        return performance_df
    results = performance_df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'csat', 'call_volume']
    for metric in metrics:
        if metric in results.columns:
            results[f'{metric}_pass'] = results[metric] <= kpis.get(metric, 600) if metric == 'aht' else results[metric] >= kpis.get(metric, 50)
    pass_columns = [f'{m}_pass' for m in metrics if f'{m}_pass' in results.columns]
    if pass_columns:
        results['overall_score'] = results[pass_columns].mean(axis=1) * 100
    return results

# Authenticate user
def authenticate_user(supabase, name, password):
    try:
        user_response = supabase.table("users").select("*").eq("name", name).execute()
        if user_response.data:
            return True, name, user_response.data[0]["role"]
        return False, None, None
    except Exception:
        return False, None, None

# Upload audio
def upload_audio(supabase, agent_name, audio_file, manager_name):
    try:
        file_name = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_file.filename}"
        res = supabase.storage.from_("call-audio").upload(file_name, audio_file.read())
        audio_url = supabase.storage.from_("call-audio").get_public_url(file_name)
        supabase.table("audio_assessments").insert({
            "agent_name": agent_name,
            "audio_url": audio_url,
            "upload_timestamp": datetime.now().isoformat(),
            "assessment_notes": "",
            "uploaded_by": manager_name
        }).execute()
        return True
    except Exception:
        return False

# Get audio assessments
def get_audio_assessments(supabase, agent_name=None):
    try:
        query = supabase.table("audio_assessments").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Update assessment notes
def update_assessment_notes(supabase, audio_id, notes):
    try:
        supabase.table("audio_assessments").update({"assessment_notes": notes}).eq("id", audio_id).execute()
        return True
    except Exception:
        return False

# Convert Plotly figure to JSON
def plotly_to_json(fig):
    if fig:
        return fig.to_json()
    return None

# Routes
@app.route('/')
def index():
    if 'user' not in session:
        return render_template('login.html')
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form.get('name')
        password = request.form.get('password')
        supabase = init_supabase()
        success, user, role = authenticate_user(supabase, name, password)
        if success:
            session['user'] = user
            session['role'] = role
            missing_critical, missing_non_critical = check_db(supabase)
            if missing_critical:
                flash(f"Critical tables missing: {', '.join(missing_critical)}.", "error")
                return redirect(url_for('login'))
            session['notifications_enabled'] = 'notifications' not in missing_non_critical
            flash("Logged in successfully!", "success")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    supabase = init_supabase()
    notifications = get_notifications(supabase).to_dict('records')
    
    if session['role'] == 'Manager':
        performance_df = get_performance(supabase)
        kpis = get_kpis(supabase)
        results = assess_performance(performance_df, kpis) if not performance_df.empty else pd.DataFrame()
        metrics = {
            'avg_overall_score': results['overall_score'].mean() if not results.empty else 0,
            'total_call_volume': results['call_volume'].sum() if not results.empty else 0,
            'agent_count': len(results['agent_name'].unique()) if not results.empty else 0
        }
        return render_template('manager_dashboard.html', notifications=notifications, metrics=metrics)
    
    elif session['role'] == 'Agent':
        performance_df = get_performance(supabase, session['user'])
        all_performance_df = get_performance(supabase)
        zoho_df = get_zoho_agent_data(supabase, session['user'])
        kpis = get_kpis(supabase)
        results = assess_performance(performance_df, kpis) if not performance_df.empty else pd.DataFrame()
        all_results = assess_performance(all_performance_df, kpis) if not all_performance_df.empty else pd.DataFrame()
        metrics = {
            'overall_score': results['overall_score'].mean() if not results.empty else 0,
            'quality_score': results['quality_score'].mean() if not results.empty else 0,
            'csat': results['csat'].mean() if not results.empty else 0,
            'attendance': results['attendance'].mean() if not results.empty else 0,
            'resolution_rate': results['resolution_rate'].mean() if not results.empty else 0,
            'contact_success_rate': results['contact_success_rate'].mean() if not results.empty else 0,
            'aht': results['aht'].mean() if not results.empty else 0,
            'talk_time': results['talk_time'].mean() if not results.empty else 0,
            'call_volume': results['call_volume'].sum() if not results.empty else 0
        }
        profile_fig = plotly_to_json(plot_performance_chart(supabase, session['user']))
        peer_fig = None
        if not all_results.empty:
            peer_avg = all_results.groupby('agent_name')['overall_score'].mean().reset_index()
            peer_avg = peer_avg[peer_avg['agent_name'] != session['user']]
            fig3 = px.box(peer_avg, y='overall_score', title="Peer Score Distribution")
            fig3.add_hline(y=metrics['overall_score'], line_dash="dash", line_color="red", annotation_text=f"Your Score: {metrics['overall_score']:.1f}%")
            peer_fig = plotly_to_json(fig3)
        tips = get_coaching_tips(supabase, session['user'])
        return render_template('agent_dashboard.html', notifications=notifications, metrics=metrics, 
                             profile_fig=profile_fig, peer_fig=peer_fig, tips=tips, user=session['user'])

@app.route('/manager/set_kpis', methods=['GET', 'POST'])
def set_kpis():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    kpis = get_kpis(supabase)
    if request.method == 'POST':
        new_kpis = {
            'attendance': float(request.form.get('attendance', kpis.get('attendance', 95.0))),
            'quality_score': float(request.form.get('quality_score', kpis.get('quality_score', 90.0))),
            'product_knowledge': float(request.form.get('product_knowledge', kpis.get('product_knowledge', 85.0))),
            'contact_success_rate': float(request.form.get('contact_success_rate', kpis.get('contact_success_rate', 80.0))),
            'onboarding': float(request.form.get('onboarding', kpis.get('onboarding', 90.0))),
            'reporting': float(request.form.get('reporting', kpis.get('reporting', 95.0))),
            'talk_time': float(request.form.get('talk_time', kpis.get('talk_time', 300.0))),
            'resolution_rate': float(request.form.get('resolution_rate', kpis.get('resolution_rate', 80.0))),
            'aht': float(request.form.get('aht', kpis.get('aht', 600.0))),
            'csat': float(request.form.get('csat', kpis.get('csat', 85.0))),
            'call_volume': int(request.form.get('call_volume', kpis.get('call_volume', 50)))
        }
        if save_kpis(supabase, new_kpis):
            flash("KPIs saved!", "success")
        else:
            flash("Error saving KPIs.", "error")
        return redirect(url_for('set_kpis'))
    return render_template('set_kpis.html', kpis=kpis)

@app.route('/manager/input_performance', methods=['GET', 'POST'])
def input_performance():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
    if request.method == 'POST':
        if 'csv_file' in request.files and request.files['csv_file']:
            csv_file = request.files['csv_file']
            df = pd.read_csv(csv_file)
            required_cols = ['agent_name', 'attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                            'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
            if all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
                    data = {col: row[col] for col in required_cols[1:]}
                    if 'date' in row:
                        data['date'] = row['date']
                    save_performance(supabase, row['agent_name'], data)
                flash(f"Imported data for {len(df)} agents!", "success")
            else:
                flash("CSV missing required columns.", "error")
        else:
            agent = request.form.get('agent')
            data = {
                'attendance': float(request.form.get('attendance', 0)),
                'quality_score': float(request.form.get('quality_score', 0)),
                'product_knowledge': float(request.form.get('product_knowledge', 0)),
                'contact_success_rate': float(request.form.get('contact_success_rate', 0)),
                'onboarding': float(request.form.get('onboarding', 0)),
                'reporting': float(request.form.get('reporting', 0)),
                'talk_time': float(request.form.get('talk_time', 0)),
                'resolution_rate': float(request.form.get('resolution_rate', 0)),
                'aht': float(request.form.get('aht', 0)),
                'csat': float(request.form.get('csat', 0)),
                'call_volume': int(request.form.get('call_volume', 0))
            }
            if save_performance(supabase, agent, data):
                flash(f"Performance saved for {agent}!", "success")
            else:
                flash("Error saving performance data.", "error")
        return redirect(url_for('input_performance'))
    return render_template('input_performance.html', agents=agents)

@app.route('/manager/assessments')
def assessments():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    performance_df = get_performance(supabase)
    kpis = get_kpis(supabase)
    results = assess_performance(performance_df, kpis) if not performance_df.empty else pd.DataFrame()
    fig = plotly_to_json(plot_performance_chart(supabase, metrics=['attendance', 'quality_score', 'csat', 'resolution_rate']))
    return render_template('assessments.html', results=results.to_dict('records') if not results.empty else [], fig=fig)

@app.route('/manager/download_assessments')
def download_assessments():
    supabase = init_supabase()
    performance_df = get_performance(supabase)
    kpis = get_kpis(supabase)
    results = assess_performance(performance_df, kpis) if not performance_df.empty else pd.DataFrame()
    csv = results.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='performance_data.csv'
    )

@app.route('/manager/set_goals', methods=['GET', 'POST'])
def set_goals():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume', 'overall_score']
    pending_goals = pd.DataFrame(supabase.table("goals").select("*").eq("status", "Awaiting Approval").in_("agent_name", agents).execute().data)
    goals_df = pd.DataFrame(supabase.table("goals").select("*").in_("agent_name", agents).execute().data)
    
    if request.method == 'POST':
        if 'single_goal' in request.form:
            agent = request.form.get('agent')
            metric = request.form.get('metric')
            target_value = float(request.form.get('target_value', 80.0))
            if set_agent_goal(supabase, agent, metric, target_value, session['user'], is_manager=True):
                flash(f"Goal set for {agent}!", "success")
        elif 'bulk_goals' in request.form:
            bulk_agents = request.form.getlist('bulk_agents')
            bulk_metric = request.form.get('bulk_metric')
            bulk_target = float(request.form.get('bulk_target', 80.0))
            for agent in bulk_agents:
                set_agent_goal(supabase, agent, bulk_metric, bulk_target, session['user'], is_manager=True)
            flash(f"Goals set for {len(bulk_agents)} agents!", "success")
        elif 'approve_goal' in request.form:
            goal_id = request.form.get('goal_id')
            if approve_goal(supabase, goal_id, session['user'], approve=True):
                flash("Goal approved!", "success")
        elif 'reject_goal' in request.form:
            goal_id = request.form.get('goal_id')
            if approve_goal(supabase, goal_id, session['user'], approve=False):
                flash("Goal rejected!", "success")
        return redirect(url_for('set_goals'))
    
    return render_template('set_goals.html', agents=agents, metrics=metrics, 
                         pending_goals=pending_goals.to_dict('records') if not pending_goals.empty else [],
                         goals=goals_df.to_dict('records') if not goals_df.empty else [])

@app.route('/manager/download_goals')
def download_goals():
    supabase = init_supabase()
    agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
    goals_df = pd.DataFrame(supabase.table("goals").select("*").in_("agent_name", agents).execute().data)
    csv = goals_df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='agent_goals.csv'
    )

@app.route('/manager/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    feedback_df = get_feedback(supabase)
    if not feedback_df.empty:
        feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        feedback_df['response_timestamp'] = pd.to_datetime(feedback_df['response_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if request.method == 'POST':
        feedback_id = request.form.get('feedback_id')
        manager_response = request.form.get('manager_response')
        if feedback_id and manager_response.strip():
            if respond_to_feedback(supabase, feedback_id, manager_response, session['user']):
                flash("Response sent!", "success")
            else:
                flash("Failed to send response.", "error")
        else:
            flash("Please provide a response and select feedback.", "error")
        return redirect(url_for('feedback'))
    
    agents = feedback_df['agent_name'].unique() if not feedback_df.empty else []
    conversations = {}
    for agent in agents:
        agent_df = feedback_df[feedback_df['agent_name'] == agent].sort_values('created_at', ascending=False)
        conversations[agent] = agent_df.to_dict('records')
    
    return render_template('feedback.html', feedback=feedback_df.to_dict('records') if not feedback_df.empty else [],
                         conversations=conversations)

@app.route('/manager/download_feedback')
def download_feedback():
    supabase = init_supabase()
    feedback_df = get_feedback(supabase)
    csv = feedback_df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='agent_feedback.csv'
    )

@app.route('/manager/audio_assessments', methods=['GET', 'POST'])
def audio_assessments():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
    audio_df = get_audio_assessments(supabase)
    if not audio_df.empty:
        audio_df['upload_timestamp'] = pd.to_datetime(audio_df['upload_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if request.method == 'POST':
        if 'audio_upload' in request.form:
            agent = request.form.get('agent')
            audio_file = request.files.get('audio_file')
            if audio_file and agent:
                if upload_audio(supabase, agent, audio_file, session['user']):
                    flash(f"Audio uploaded for {agent}!", "success")
                else:
                    flash("Failed to upload audio.", "error")
            else:
                flash("Please select an agent and audio file.", "error")
        elif 'save_notes' in request.form:
            audio_id = request.form.get('audio_id')
            notes = request.form.get('notes')
            if update_assessment_notes(supabase, audio_id, notes):
                flash("Notes saved!", "success")
            else:
                flash("Failed to save notes.", "error")
        return redirect(url_for('audio_assessments'))
    
    return render_template('audio_assessments.html', agents=agents, audio_assessments=audio_df.to_dict('records') if not audio_df.empty else [])

@app.route('/manager/download_audio_assessments')
def download_audio_assessments():
    supabase = init_supabase()
    audio_df = get_audio_assessments(supabase)
    csv = audio_df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='audio_assessments.csv'
    )

@app.route('/manager/leaderboard', methods=['GET', 'POST'])
def leaderboard():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    leaderboard_df = get_leaderboard(supabase)
    fig = None
    if not leaderboard_df.empty:
        fig = px.bar(leaderboard_df, x="agent_name", y="overall_score", color="agent_name", title="Agent Leaderboard")
    
    agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
    if request.method == 'POST':
        agent = request.form.get('agent')
        badge_name = request.form.get('badge_name')
        description = request.form.get('description')
        if award_badge(supabase, agent, badge_name, description, session['user']):
            flash(f"Badge awarded to {agent}!", "success")
        else:
            flash("Error awarding badge.", "error")
        return redirect(url_for('leaderboard'))
    
    return render_template('leaderboard.html', leaderboard=leaderboard_df.to_dict('records') if not leaderboard_df.empty else [],
                         fig=plotly_to_json(fig), agents=agents)

@app.route('/manager/community_forum', methods=['GET', 'POST'])
def manager_community_forum():
    if 'user' not in session or session['role'] != 'Manager':
        return redirect(url_for('login'))
    supabase = init_supabase()
    category = request.form.get('category', 'Tips') if request.method == 'POST' else request.args.get('category', 'Tips')
    if request.method == 'POST':
        message = request.form.get('message')
        if message and create_forum_post(supabase, session['user'], message, category):
            flash("Post submitted!", "success")
        else:
            flash("Error creating forum post.", "error")
        return redirect(url_for('manager_community_forum', category=category))
    
    posts_df = get_forum_posts(supabase, category)
    return render_template('community_forum.html', posts=posts_df.to_dict('records') if not posts_df.empty else [],
                         category=category, role='Manager')

@app.route('/agent/goals', methods=['GET', 'POST'])
def agent_goals():
    if 'user' not in session or session['role'] != 'Agent':
        return redirect(url_for('login'))
    supabase = init_supabase()
    performance_df = get_performance(supabase, session['user'])
    kpis = get_kpis(supabase)
    results = assess_performance(performance_df, kpis) if not performance_df.empty else pd.DataFrame()
    goals_df = pd.DataFrame(supabase.table("goals").select("*").eq("agent_name", session['user']).execute().data)
    
    if request.method == 'POST':
        metric = request.form.get('metric')
        target_value = float(request.form.get('target_value', 80.0))
        if set_agent_goal(supabase, session['user'], metric, target_value, session['user'], is_manager=False):
            flash(f"Goal submitted for {metric}! Awaiting manager approval.", "success")
        else:
            flash("Error setting goal.", "error")
        return redirect(url_for('agent_goals'))
    
    all_metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                   'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume', 'overall_score']
    goals_data = []
    for metric in all_metrics:
        goal_row = goals_df[goals_df['metric'] == metric]
        current_value = results[results['date'] == max(results['date'])][metric].mean() if not results.empty and metric in results.columns else 0.0
        if not goal_row.empty:
            row = goal_row.iloc[0]
            progress = min((kpis.get(metric, 600) - current_value) / (kpis.get(metric, 600) - row['target_value']) * 100, 100) if metric == 'aht' else min(current_value / row['target_value'] * 100, 100) if row['target_value'] > 0 else 0
            color = "bg-success" if progress >= 80 else "bg-warning" if progress >= 50 else "bg-danger"
            goals_data.append({
                'metric': metric,
                'target_value': row['target_value'],
                'current_value': current_value,
                'status': row['status'],
                'progress': progress,
                'progress_color': color
            })
        else:
            goals_data.append({'metric': metric, 'target_value': None, 'current_value': current_value, 'status': None, 'progress': 0, 'progress_color': 'bg-secondary'})
    
    return render_template('agent_goals.html', goals=goals_data, metrics=all_metrics)

@app.route('/agent/feedback', methods=['GET', 'POST'])
def agent_feedback():
    if 'user' not in session or session['role'] != 'Agent':
        return redirect(url_for('login'))
    supabase = init_supabase()
    if request.method == 'POST':
        feedback_text = request.form.get('feedback_text')
        if feedback_text:
            supabase.table("feedback").insert({
                "agent_name": session['user'],
                "message": feedback_text
            }).execute()
            if session.get("notifications_enabled", False):
                managers = supabase.table("users").select("id").eq("role", "Manager").execute()
                for manager in managers.data:
                    supabase.table("notifications").insert({
                        "user_id": manager["id"],
                        "message": f"New feedback from {session['user']}: {feedback_text[:50]}..."
                    }).execute()
            flash("Feedback submitted!", "success")
        else:
            flash("Please enter feedback.", "error")
        return redirect(url_for('agent_feedback'))
    
    feedback_df = get_feedback(supabase, session['user'])
    if not feedback_df.empty:
        feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        feedback_df['response_timestamp'] = pd.to_datetime(feedback_df['response_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('agent_feedback.html', feedback=feedback_df.to_dict('records') if not feedback_df.empty else [])

@app.route('/agent/download_feedback')
def agent_download_feedback():
    supabase = init_supabase()
    feedback_df = get_feedback(supabase, session['user'])
    csv = feedback_df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='feedback_history.csv'
    )

@app.route('/agent/tickets')
def agent_tickets():
    if 'user' not in session or session['role'] != 'Agent':
        return redirect(url_for('login'))
    supabase = init_supabase()
    zoho_df = get_zoho_agent_data(supabase, session['user'])
    total_tickets = zoho_df['id'].nunique() if not zoho_df.empty else 0
    channel_counts = zoho_df.groupby('channel')['id'].nunique().reset_index(name='Ticket Count') if not zoho_df.empty else pd.DataFrame()
    fig = None
    if not channel_counts.empty:
        fig = px.pie(channel_counts, values='Ticket Count', names='channel', title="Ticket Distribution by Channel")
    
    if not zoho_df.empty:
        time_col = 'created_time' if 'created_time' in zoho_df.columns else 'created_at' if 'created_at' in zoho_df.columns else None
        if time_col:
            zoho_df[time_col] = pd.to_datetime(zoho_df[time_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('agent_tickets.html', tickets=zoho_df.to_dict('records') if not zoho_df.empty else [],
                         total_tickets=total_tickets, channel_counts=channel_counts.to_dict('records') if not channel_counts.empty else [],
                         fig=plotly_to_json(fig))

@app.route('/agent/download_tickets')
def agent_download_tickets():
    supabase = init_supabase()
    zoho_df = get_zoho_agent_data(supabase, session['user'])
    csv = zoho_df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='zoho_agent_data.csv'
    )

@app.route('/agent/achievements')
def agent_achievements():
    if 'user' not in session or session['role'] != 'Agent':
        return redirect(url_for('login'))
    supabase = init_supabase()
    badges_df = pd.DataFrame(supabase.table("badges").select("*").eq("agent_name", session['user']).execute().data)
    return render_template('agent_achievements.html', badges=badges_df.to_dict('records') if not badges_df.empty else [])

@app.route('/agent/community_forum', methods=['GET', 'POST'])
def agent_community_forum():
    if 'user' not in session or session['role'] != 'Agent':
        return redirect(url_for('login'))
    supabase = init_supabase()
    category = request.form.get('category', 'Tips') if request.method == 'POST' else request.args.get('category', 'Tips')
    if request.method == 'POST':
        message = request.form.get('message')
        if message and create_forum_post(supabase, session['user'], message, category):
            flash("Post submitted!", "success")
        else:
            flash("Error creating forum post.", "error")
        return redirect(url_for('agent_community_forum', category=category))
    
    posts_df = get_forum_posts(supabase, category)
    return render_template('community_forum.html', posts=posts_df.to_dict('records') if not posts_df.empty else [],
                         category=category, role='Agent')

@app.route('/agent/ask_coach', methods=['GET', 'POST'])
def ask_coach_route():
    if 'user' not in session or session['role'] != 'Agent':
        return redirect(url_for('login'))
    supabase = init_supabase()
    answer = None
    if request.method == 'POST':
        question = request.form.get('question')
        if question.strip():
            answer = ask_coach(supabase, session['user'], question)
        else:
            flash("Please enter a question.", "error")
    return render_template('ask_coach.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
