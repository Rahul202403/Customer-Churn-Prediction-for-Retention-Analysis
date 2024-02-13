from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

load_model = pickle.load(open('prediction_model.sav','rb'))
segmentation_model = pickle.load(open('segmentation_model.sav','rb'))
df = pd.read_csv('Clustered_Customer_Data.csv')

app = Flask(__name__)

def return_values(value):
    if value == 'yes':
        return 1
    else:
        return 0

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction')
def prediction():
    return render_template("churn_prediction.html")

@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    if request.method == 'POST':
        customerId = request.form['customerId']
        gender = request.form['gender']
        if gender == 'male':
            gender = 1
        else:
            gender = 0
        sr_citizen = request.form['sr_citizen']
        sr_citizen = return_values(sr_citizen)
        partner = request.form['partner']
        partner = return_values(partner)
        dependents = request.form['dependents']
        dependents = return_values(dependents)
        tenure = int(request.form['tenure'])
        phone_service = request.form['phone_service']
        phone_service = return_values(phone_service)
        multiple_lines = request.form['multiple_lines']
        multiple_lines = return_values(multiple_lines)
        online_security = request.form['online_security']
        online_security = return_values(online_security)
        online_backup = request.form['online_backup']
        online_backup = return_values(online_backup)
        device_protection = request.form['device_protection']
        device_protection = return_values(device_protection)
        tech_support = request.form['tech_support']
        tech_support = return_values(tech_support)
        streaming_tv = request.form['streaming_tv']
        streaming_tv = return_values(streaming_tv)
        streaming_movies = request.form['streaming_movies']
        streaming_movies = return_values(streaming_movies)
        paperless_billing = request.form['paperless_billing']
        paperless_billing = return_values(paperless_billing)
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        internet_service = request.form['internet_service']
        if internet_service == 'dsl':
            is_dsl = 1
            is_fiber_optic = 0
            is_no = 0
        elif internet_service == 'fiber_optic':
            is_dsl = 0
            is_fiber_optic = 1
            is_no = 0
        else:
            is_dsl = 0
            is_fiber_optic = 0
            is_no = 1
        contract = request.form['contract']
        if contract == 'month-to-month':
            contract_m2m = 1
            contract1yr = 0
            contract2yr = 0
        elif contract == 'one_year':
            contract_m2m = 0
            contract1yr = 1
            contract2yr = 0
        else:
            contract_m2m = 0
            contract1yr = 0
            contract2yr = 1
        payment_method = request.form['payment_method']
        if payment_method == 'bank_transfer':
            bank_transfer = 1
            credit_card = 0
            elec_check = 0
            mail_check = 0
        elif payment_method == 'credit_card':
            bank_transfer = 0
            credit_card = 1
            elec_check = 0
            mail_check = 0
        elif payment_method == 'elec_check':
            bank_transfer = 0
            credit_card = 0
            elec_check = 1
            mail_check = 0
        else:
            bank_transfer = 0
            credit_card = 0
            elec_check = 0
            mail_check = 1
        input = [gender,sr_citizen,partner,dependents,tenure,phone_service,multiple_lines,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,paperless_billing,monthly_charges,total_charges,is_dsl,is_fiber_optic,is_no,contract_m2m,contract1yr,contract2yr,bank_transfer,credit_card,elec_check,mail_check]
        print(input,customerId)
        
        churn = load_model.predict([input])
        churn = churn[0]
        print(churn)

        segmentation_input = [gender,sr_citizen,partner,dependents,tenure,phone_service,multiple_lines,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,paperless_billing,monthly_charges,total_charges,churn,is_dsl,is_fiber_optic,is_no,contract_m2m,contract1yr,contract2yr,bank_transfer,credit_card,elec_check,mail_check]
        clust = segmentation_model.predict([segmentation_input])
        clust = clust[0]
        print(clust)
        if clust == 0:
            churn_time = abs(30-tenure)
        elif clust == 1:
            churn_time = abs(59-tenure)
        elif clust == 2:
            churn_time = abs(20-tenure)
        else:
            churn_time = abs(18-tenure)

        cluster_df1 = df[df['Cluster']==clust]
        plt.rcParams["figure.figsize"] = (20,3)
        i = 0
        for c in cluster_df1.drop(['Cluster'],axis=1):
            fig, ax = plt.subplots()
            grid = sns.FacetGrid(cluster_df1, col='Cluster',col_wrap=3)
            grid.map(plt.hist, c)
            # plt.show()
            # strFile = r"C:\Users\lenovo\OneDrive\Desktop\Customer Churn Prediction for Retention Analysis\static\images\plot.png"
            strFile = os.getcwd()+f"\static\images\plot{i}.png"
            i = i + 1
            if os.path.isfile(strFile):
                os.remove(strFile)
            grid.savefig(strFile)

        # Retention Strategies Recommendation
        cluster_0_strategies = ['Create educational content or campaigns targeting the importance of online security, device protection, and the benefits of online backup.',
                                'Provide easy-to-understand guides on setting up security features and using online backup services.',
                                'Highlight the risks of not having proper online security, backup, and device protection to emphasize their importance.',
                                'Introduce limited-time promotions or discounts for online security, device protection, and tech support services. This could incentivize users to subscribe to these services.',
                                'Send personalized messages and offers that align with their usage patterns and preferences.',
                                'Use push notifications to remind them of exclusive deals, discounts, or promotions that cater to their interests.',
                                'Promote the availability of tech support services and emphasize the benefits of having a reliable support system in case of any issues.',
                                'Offer tutorials or guides on how to troubleshoot common problems they may encounter with their devices or services.',
                                'Provide special discounts or free trials for online security and backup services to encourage adoption.',
                                'Introduce affordable device protection plans and highlight the cost-effectiveness compared to potential device replacement costs.',
                                'Illustrate real-life scenarios where device protection could have saved them money and inconvenience.',
                                'Introduce curated content bundles or partnerships with streaming services to add value for users not currently engaged in streaming.',
                                'Emphasize the entertainment benefits of streaming to encourage adoption and increase overall satisfaction.',
                                'Continue to offer paperless billing options but provide additional incentives for users to switch to electronic billing.',
                                'Implement user-friendly interfaces for electronic billing and provide step-by-step guides to make the transition seamless.',
                                'Highlight the advantages of fiber optic internet, such as faster speeds, reliable connections, and improved online experiences.',
                                'Provide ongoing updates on network improvements or upgrades to showcase a commitment to delivering quality services.',
                                'Emphasize the flexibility of the month-to-month contract, allowing users to adapt their plans as their needs change.',
                                'Consider offering loyalty rewards or discounts for customers who remain on month-to-month contracts for an extended period.',
                                'Provide incentives such as discounts or reward points for users who opt for electronic check payment methods.',
                                'Streamline the electronic payment process to make it quick, secure, and user-friendly.',
                                'Implement a loyalty program that rewards customers for their tenure and engagement. Offer discounts, exclusive content, or special promotions to long-term users.']
        cluster_1_strategies = ['Launch targeted campaigns to educate users about online security risks. Offer simple and practical tips for securing their devices and personal information.',
                                'Introduce a bundled package that includes basic online security, device protection, and tech support services at a discounted rate. Highlight the added value and convenience for customers.',
                                'Encourage users to set up online backups for their data by offering a limited-time promotion or additional storage for free. Emphasize the importance of safeguarding important information.',
                                'Develop a loyalty program that rewards customers for their tenure. Offer exclusive discounts, perks, or personalized deals to long-term users, creating a sense of appreciation and value.',
                                'While the customer may not prefer paperless billing, offer incentives such as discounts, reward points, or exclusive promotions to encourage the transition to digital billing. Emphasize the environmental benefits and convenience.',
                                'Provide more payment options beyond mailed checks, such as online payment portals or mobile payment apps. Ensure that the transition is seamless and accompanied by clear instructions.',
                                'Stay in regular contact through non-intrusive channels like email or SMS. Share useful tips, service updates, and exclusive offers to keep users engaged and informed about the value of their mobile service.',
                                'Invest in customer support services that are easily accessible and tailored to individual needs. Ensure a hassle-free experience for users seeking assistance with any issues.',
                                'If applicable, consider simplifying the billing structure to make it more transparent. Clearly communicate the charges and services rendered to avoid any confusion or dissatisfaction.',
                                'Occasionally surprise customers with small gestures of appreciation, such as discounts, freebies, or personalized thank-you messages. This helps in building a positive emotional connection with the brand.',
                                'Establish a feedback mechanism to understand customer satisfaction levels. Use the insights gained to continuously improve services and address any pain points.']
        cluster_2_strategies = ['Provide personalized discounts, loyalty rewards, or exclusive offers based on their usage patterns and preferences.',
                                'Offer bundling discounts for multiple lines or services, such as combining mobile and internet services.',
                                'Regularly communicate with customers through targeted newsletters, emails, or in-app notifications to keep them informed about new features, promotions, or upcoming events.',
                                'Conduct customer surveys to gather feedback and show that their opinions are valued.',
                                'Provide educational content on the importance of online security, backup practices, and device protection to enhance their awareness and utilization of existing services.',
                                'Offer tutorials or workshops on maximizing the benefits of streaming services, online security features, and other value-added services.',
                                'Offer exclusive content or early access to new services for long-tenured customers to make them feel special.',
                                'Collaborate with streaming services to provide exclusive content or discounts for your customers.',
                                'Ensure a seamless and paperless billing process, and offer additional incentives for customers who choose paperless billing.',
                                'Provide flexibility in payment methods and consider offering discounts for customers who use bank transfers or credit cards',
                                'Offer priority access to dedicated customer support for high-value customers to resolve any issues promptly.',
                                'Provide tech support and troubleshooting assistance for their internet services to enhance their overall experience.',
                                'Provide attractive incentives for customers who renew their contracts for 1 or 2 years, such as additional discounts or bonus services.',
                                'Keep customers informed about upcoming contract renewals and provide reminders of the benefits of continuing with your services.',
                                'Foster a sense of community by organizing customer events, forums, or online groups where users can share experiences and tips.',
                                'Encourage customer referrals by offering rewards for successful referrals, thereby expanding your customer base.',
                                'Continuously improve the quality of your services, addressing any potential issues promptly.',
                                'Proactively inform customers about network upgrades, maintenance schedules, or improvements to assure them of your commitment to service quality.',
                                'Monitor the market for competitive pricing and ensure that your pricing remains competitive while offering superior value.',
                                'Emphasize the overall value of your bundled services, including internet, streaming, and mobile, to justify the higher monthly charges.']
        cluster_3_strategies = ['Provide targeted education on online security to address the lack of online security measures. Send emails or text messages with tips on creating strong passwords, enabling two-factor authentication, and recognizing phishing attempts.',
                                'Offer affordable add-ons for online security, backup, and device protection. Create a bundle that includes these services at a discounted rate, specifically designed for users with moderate monthly charges.',
                                'Establish a dedicated tech support channel for users who are not tech-savvy. Provide assistance through chat or phone for any issues they might face with their devices, internet services, or mobile plans.',
                                'Encourage the use of electronic check payments by offering exclusive promotions or discounts for users who opt for this payment method. This can be a cost-effective way to incentivize customer loyalty.',
                                'Create exclusive promotions or discounts for DSL internet services to enhance the overall value for customers. Offer free upgrades, speed boosts, or other perks to keep them satisfied with their internet connection.',
                                'Consider introducing loyalty-based contract benefits, such as reduced rates or additional features for customers who commit to a longer-term contract. This can be a way to encourage users to extend their commitment.',
                                'Conduct periodic account reviews to understand users evolving needs. Reach out to them with personalized recommendations, ensuring they are aware of relevant services and features that may enhance their mobile and internet experience.',
                                'Send surveys to understand users communication preferences. Tailor your outreach based on their preferred channels and frequency, ensuring that they receive information and updates in a manner that suits them.',
                                'Implement a reward program that recognizes and appreciates long-term customers. Offer exclusive perks, discounts, or freebies based on their tenure with your mobile and internet services.',
                                'Establish a feedback loop where customers can share their opinions and concerns easily. Actively use this feedback to address issues and make improvements, demonstrating your commitment to customer satisfaction.']
        if churn == 1:
            churn_message = 'Potential Churner'
            churn_time_message = f"Likely to churn in {churn_time} months"
        elif churn == 0:
            churn_message = 'Loyal Customer (Not a Churner)'
            churn_time_message = 'None'

        if clust == 0:
            type = 'Unsatisfied Churner (Cluster 0)'
            retention_strategies = cluster_0_strategies
        elif clust == 1:
            type = 'Conditional Churner (Cluster 1)'
            retention_strategies = cluster_1_strategies
        elif clust == 2:
            type = 'Conditionally Loyal Subscriber (Cluster 2)'
            retention_strategies = cluster_2_strategies
        else:
            type = 'Lifestyle Migrator (Cluster 3)'
            retention_strategies = cluster_3_strategies
        
        if gender == 1:
            image = '../static/images/profile_male.jpg'
        else:
            image = '../static/images/profile_female.jpg'

    return render_template("results.html",churn_message = churn_message,time_period = churn_time_message,segment_type = type,retention_strategies = retention_strategies, customerId = customerId,profile_image = image)

if __name__ == '__main__':
    app.run(debug=True)