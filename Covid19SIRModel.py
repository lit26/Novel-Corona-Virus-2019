#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:29:04 2020

@author: tianningli
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from tqdm import tqdm_notebook

class Covid19SIRModel:
    def __init__(self, country, N):
        self.country = country
        self.df_trend = []
        self.n = 0
        self.N = N
        self.t = []
        self.beta = 0
        self.gamma = 0
        self.delta = 0
        self.s0 = 0
        self.i0 = 0
        self.r0 = 0
        self.d0 = 0

    def timeSpan(self):
        return self.t
    
    def SIR_model(self, y, t, N, beta, gamma, delta):
        s, i, r, d = y
        ds_dt = -beta * s * i 
        di_dt = beta * s * i  - gamma * i - delta * i
        dr_dt = gamma * i
        dd_dt = delta * i
        return ds_dt, di_dt, dr_dt, dd_dt
    
    def readFile(self,base_path, file,choice):
    	df= pd.read_csv(base_path+file)
    	df.drop(["Lat","Long"], axis=1, inplace=True)
    	df=df.melt(id_vars=["Country/Region","Province/State"], 
    	        var_name="Date", 
    	        value_name=choice)
    	df['Date'] = pd.to_datetime(df['Date'])
    	df['Date']=df['Date'].apply(lambda x:x.date())
    	return df
    
    def readData(self):
        base_path = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
        df_confirmed = self.readFile(base_path, 'time_series_covid19_confirmed_global.csv','Confirmed')
        df_deaths = self.readFile(base_path, 'time_series_covid19_deaths_global.csv','Deaths')
        df_recovered = self.readFile(base_path, 'time_series_covid19_recovered_global.csv','Recovered')
        df_confirmed = df_confirmed[df_confirmed['Country/Region']==self.country]
        df_deaths = df_deaths[df_deaths['Country/Region']==self.country]
        df_recovered = df_recovered[df_recovered['Country/Region']==self.country]
        df = df_confirmed.append(df_deaths)
        df = df.append(df_recovered)
        df = df[['Date', 'Province/State', 'Country/Region', 'Confirmed','Deaths','Recovered']]
        df[['Confirmed','Deaths','Recovered']] = df[['Confirmed','Deaths','Recovered']].fillna(0)
        self.df_trend = df.groupby(['Date']).sum()
        self.df_trend['Infected'] = self.df_trend['Confirmed']-self.df_trend['Deaths']-self.df_trend['Recovered']
        self.df_trend['Death_rate'] = self.df_trend.apply(lambda x: x['Deaths']*100/(x['Confirmed']), axis=1)
        self.df_trend = self.df_trend.reset_index()
        self.n = len(self.df_trend)
        start_date = self.df_trend['Date'][0].strftime("%Y-%m-%d")
        end_date = self.df_trend['Date'][self.n - 1].strftime("%Y-%m-%d")
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        self.t = np.linspace(start.value, end.value, self.n)
        self.t = pd.to_datetime(self.t)
        return self.df_trend
    
    def plotRealData(self):
        plt.plot(self.t, 'Confirmed', data=self.df_trend, color='#D79913')
        plt.plot(self.t, 'Deaths', data=self.df_trend, color='red')
        plt.plot(self.t, 'Recovered', data=self.df_trend, color='green')
        plt.ylabel('Number of cases', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        plt.plot(self.t, 'Infected', data=self.df_trend, color='#D79913')
        plt.plot(self.t, 'Recovered', data=self.df_trend, color='green')
        plt.ylabel('Number of cases', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        
    def setUpInitialCondition(self):
        I0, R0, D0 = self.df_trend['Infected'][0], self.df_trend['Recovered'][0], self.df_trend['Deaths'][0]
        S0 = self.N - I0 - R0 - D0
        self.s0 = S0 / self.N
        self.i0 = I0 / self.N
        self.r0 = R0 / self.N
        self.d0 = D0 / self.N
    
    def calculateRates(self, rate, choice):
        rate_list = [0]
        for i in range(1,self.n):
            if rate == "gamma":
                rate_list.append((self.df_trend["Recovered"][i] - self.df_trend["Recovered"][i-1]) / self.df_trend["Infected"][i-1])
            elif rate == "delta":
                rate_list.append((self.df_trend["Deaths"][i] - self.df_trend["Deaths"][i-1]) / self.df_trend["Infected"][i-1])
            elif rate == "beta":
                I_prev = self.df_trend["Infected"][i-1]
                S = self.N - self.df_trend["Infected"][i] - self.df_trend["Recovered"][i]
                S_prev = self.N - I_prev - self.df_trend["Recovered"][i-1]
                rate_list.append((self.N*(S_prev-S))/(S_prev*I_prev))
        if choice == 'Recovered':
            color = 'green'
        elif choice == 'Infected':
            color = '#D79913'
        elif choice == 'Deaths':
            color = 'red'
        plt.plot(list(self.t), rate_list, color=color)
        plt.ylabel(choice, fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.xticks(rotation=20)
        plt.show()
        return rate_list
    
    def calculateHyperParameter(self):
        # calculate gamma
        self.gamma = self.df_trend["Recovered"][self.n-1] / sum(self.df_trend["Infected"])
        print("Gamma: ", self.gamma)
        
        # calculate delta
        self.delta = self.df_trend["Deaths"][self.n-1] / sum(self.df_trend["Infected"])
        print("Delta: ", self.delta)
        
        # tune for beta
        MSE = []
        for beta in tqdm_notebook([i/1000 for i in range(1,1000)]):
            y0 = self.s0, self.i0, self.r0, self.d0
            days=np.linspace(0,self.n-1,self.n)
            ret = scipy.integrate.odeint(self.SIR_model, y0, days, args=(self.N, beta, self.gamma, self.delta))
            S, I, R, D = ret.T * self.N 
            error_infected = sum(abs(self.df_trend['Infected']-I))
            error_recovered = sum(abs(self.df_trend['Recovered']-R))
            error_deaths = sum(abs(self.df_trend['Deaths']-D))
            MSE.append(error_infected + error_recovered + error_deaths)
        self.beta = MSE.index(min(MSE))/1000
        print("Beta: ", self.beta)
        return self.beta, self.gamma, self.delta
    
    def setParameter(self, choice, rate):
        if choice == 'beta':
            self.beta = rate
        elif choice == 'gamma':
            self.gamma = rate
        elif choice == 'delta':
            self.delta = rate
        
    def modelling(self,shift_factor):
        y0 = self.s0, self.i0, self.r0, self.d0
        days=np.linspace(0,self.n-1,self.n)
        ret = scipy.integrate.odeint(self.SIR_model, y0, days, args=(self.N, self.beta, self.gamma, self.delta))
        S, I, R, D = ret.T * self.N
        
        for i in range(shift_factor):
            S = [0]+list(S)
            S = S[:-1]
            I = [0]+list(I)
            I = I[:-1]
            R = [0]+list(R)
            R = R[:-1]
            D = [0]+list(D)
            D = D[:-1]
        
        # Plot
        plt.figure(figsize=[8,6])
        plt.plot(self.t, I, color='#D79913', label='Infected_SIRmodel')
        plt.plot(self.t, 'Infected', data=self.df_trend, linestyle='dashed',color='#D79913',label='Infected_real')
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        plt.figure(figsize=[8,6])
        plt.plot(self.t, R, color='g', label='Recovered_SIRmodel')
        plt.plot(self.t, 'Recovered', data=self.df_trend, linestyle='dashed',color='g',label='Recovered_real')
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        plt.figure(figsize=[8,6])
        plt.plot(self.t, D, color='r', label='Deaths_SIRmodel')
        plt.plot(self.t, 'Deaths', data=self.df_trend, linestyle='dashed',color='r',label='Deaths_real')
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        plt.figure(figsize=[8,6])
        plt.plot(self.t, I, color='#D79913', label='Infected_SIRmodel')
        plt.plot(self.t, 'Infected', data=self.df_trend, linestyle='dashed',color='#D79913',label='Infected_real')
        plt.plot(self.t, R, color='g', label='Recovered_SIRmodel')
        plt.plot(self.t, 'Recovered', data=self.df_trend, linestyle='dashed',color='g',label='Recovered_real')
        plt.plot(self.t, D, color='r', label='Deaths_SIRmodel')
        plt.plot(self.t, 'Deaths', data=self.df_trend, linestyle='dashed',color='r',label='Deaths_real')
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        return S, I, R, D
    
    def prediction(self, start, end, shift_factor):
        s = [int(i) for i in start.split("-")]
        e = [int(i) for i in end.split("-")]
        d0 = date(s[0],s[1],s[2])
        d1 = date(e[0],e[1],e[2])
        delta = d1 - d0
        ndays = delta.days + 1
        
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        t = np.linspace(start.value, end.value, ndays)
        t = pd.to_datetime(t)
        y0 = self.s0, self.i0, self.r0, self.d0
        days = np.linspace(0,ndays-1,ndays)
        prediction = scipy.integrate.odeint(self.SIR_model, y0, days, args=(self.N, self.beta, self.gamma, self.delta))
        S_p, I_p, R_p, D_p = prediction.T * self.N
        for i in range(shift_factor):
            S_p = [0]+list(S_p)
            S_p = S_p[:-1]
            I_p = [0]+list(I_p)
            I_p = I_p[:-1]
            R_p = [0]+list(R_p)
            R_p = R_p[:-1]
            D_p = [0]+list(D_p)
            D_p = D_p[:-1]
        plt.figure(figsize=[8,6])
        plt.plot(t, I_p, color='#D79913', label='Infected_pred')
        plt.plot(t, R_p, color='g', label='Recovered_pred')
        plt.plot(t, D_p, color='r', label='Deaths_pred')
        plt.xticks(rotation=20)
        plt.legend()
        plt.show()
        return t, S_p, I_p, R_p, D_p

        