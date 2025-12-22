import streamlit as st
import pickle 
import pandas as pd
import numpy as np

with open('socialmedia-productivity.pkl','rb') as obj1:
    data=pickle.load(obj1)
st.title("📘 Analyzing the Impact of Digital Habits on Productivity")
st.caption("A machine learning application exploring how social media usage, digital wellbeing, and lifestyle factors influence productivity.")
st.markdown("---")
st.write("Enter the details below:")

def improvement_messages(
        sleep_hours,
        stress_level,
        screen_time_before_sleep,
        daily_social_media_time,
        breaks_during_work,
        coffee_consumption_per_day,
        weekly_offline_hours):
        messages = []
        if sleep_hours < 6:
            messages.append("😴 Increase your sleep to at least 7–8 hours for better focus.")
        else:
            messages.append("✅ Your sleep duration looks healthy.")
        if stress_level > 5:
            messages.append("🧠 High stress detected. Try short breaks, meditation, or light exercise.")
        else:
            messages.append("✅ Your stress level is within a manageable range.")
        if screen_time_before_sleep > 1:
            messages.append("📱 Reduce screen time before sleep to improve sleep quality.")
        else:
            messages.append("✅ Screen time before sleep is well controlled.")
        if daily_social_media_time > 3:
            messages.append("⏳ Limit social media usage during work hours.")
        else:
            messages.append("✅ Social media usage is within a reasonable limit.")
        if breaks_during_work < 3:
            messages.append("⏸️ Taking short breaks can significantly improve productivity.")
        else:
            messages.append("✅ You are taking adequate breaks during work.")
        if coffee_consumption_per_day > 3:
            messages.append("☕ Too much caffeine may increase stress and reduce sleep quality.")
        else:
            messages.append("✅ Coffee consumption is within a healthy range.")
        if weekly_offline_hours < 10:
            messages.append("🌿 Spend more offline time to reduce burnout.")
        else:
            messages.append("✅ You maintain a good balance between online and offline time.")
        return messages

age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
job_type = st.selectbox("Job Type",['Education','IT','Finance','Student','Unemployed','Health'])
daily_social_media_time = st.slider("Daily Social Media Time (hours)", 0.000, 24.000, 2.000)
social_platform_preference = st.selectbox("Preferred Social Platform",['Facebook', 'Twitter', 'Telegram', 'TikTok', 'Instagram'])
number_of_notifications = st.number_input("Number of Notifications per Day", min_value=0, max_value=150, value=50)
work_hours_per_day = st.slider("Work Hours per Day", 0.000, 16.000, 5.000)
stress_level = st.slider("Stress Level (0–10)", 0.0, 10.0, 2.0)
sleep_hours = st.slider("Sleep Hours per Day", 0.000, 12.0, 3.000)
screen_time_before_sleep = st.slider("Screen Time Before Sleep (hours)", 0.000, 6.000, 1.000)
breaks_during_work = st.number_input("Breaks During Work (per day)", min_value=0, max_value=20, value=3)
uses_focus_apps_val = st.checkbox("Uses Focus Apps")
has_digital_wellbeing_enabled_val = st.checkbox("Digital Wellbeing Enabled")
uses_focus_apps= int(uses_focus_apps_val)
has_digital_wellbeing= int(has_digital_wellbeing_enabled_val)
coffee_consumption_per_day = st.number_input("Coffee Consumption per Day", min_value=0, max_value=20, value=2)
days_feeling_burnout_per_month = st.slider("Days Feeling Burnout per Month", 0, 31, 3)
weekly_offline_hours = st.slider("Weekly Offline Hours", 0.000, 50.000, 5.000)
job_satisfaction_score = st.slider("Job Satisfaction Score (0–10)", 0.000, 10.000, 2.000)

button=st.button('Predict')
if button: 
    numeric_features=np.array([[age, daily_social_media_time, number_of_notifications,
       work_hours_per_day,stress_level, sleep_hours,
       screen_time_before_sleep, breaks_during_work, coffee_consumption_per_day,
       days_feeling_burnout_per_month, weekly_offline_hours,
       job_satisfaction_score,uses_focus_apps,has_digital_wellbeing]])
    cat_df = pd.DataFrame([[gender,job_type, social_platform_preference]],
             columns=data['onehot'].feature_names_in_)
    onehot_features=data['onehot'].transform(cat_df)
    final_input=np.hstack([numeric_features,onehot_features]) 
    final_df = pd.DataFrame(final_input,columns=data['scaler'].feature_names_in_)
    scaled_input = data['scaler'].transform(final_df)
    result=data['model'].predict(scaled_input)[0]
    # st.write('Raw model output:', result)
    st.success(f"📈 Predicted Productivity Score: **{result:.3f}**")
    if result >= 7.5:
        st.info("🌟 Excellent productivity! Your digital habits appear well balanced. Keep maintaining these healthy patterns.")
    elif result >= 4:
        st.warning("⚖️ Moderate productivity. There is room for improvement by optimizing digital habits and work routines.")
    else:
        st.error("⚠️ Low productivity detected. Consider reducing digital distractions, improving sleep, and managing stress levels.")

    messages = improvement_messages(
    sleep_hours,
    stress_level,
    screen_time_before_sleep,
    daily_social_media_time,
    breaks_during_work,
    coffee_consumption_per_day,
    weekly_offline_hours)
    st.subheader("🛠️ Productivity Feedback")
    for msg in messages:
        if "✅" in msg:
            st.success(msg)
            # pass
        else:
            st.warning(msg)
    if not messages:
        messages.append("✅ Your habits look well balanced. Keep it up!")

    if all("✅" in msg for msg in messages):
            st.success("🎉 Great job! Your habits strongly support high productivity.")
    good_count = sum("✅" in msg for msg in messages)
    if good_count >= len(messages) - 1:
        st.success("👍 You are doing great overall. Only minor improvements needed.")

