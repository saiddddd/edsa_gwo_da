# Creator : Said Al Afghani Edsa
# Date : 7/10/23
#
#
# can be used as features selection, etc...
import numpy as np

np.random.seed(42)


def binary_GWO_da(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    Convergence_curve=np.zeros(Max_iter)

     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    

    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
            
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        a_da=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=((X1)+(X2)+(X3))/3     
        #Convergence_curve[l]=Alpha_score;
         # DA Update
                for j in range(0,dim):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    r3 = np.random.randint(SearchAgents_no)
                    LEVY = np.random.standard_cauchy()
                    
                Positions[i][j] += (a_da * r1 * (Alpha_pos[j] - Positions[i][j]) +
                                          a_da * r2 * (Positions[r3][j] - Positions[i][j]) +
                                          LEVY)                    
              

                
#                 #sigmoid function
#                 s = 1/(1 + np.exp(-Positions[i][j]))
                
#                 r = np.random.rand()
#                 if s > r:
#                     Positions[i][j] = 1
#                 else:
#                     Positions[i][j] = 0

        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
    
    print(Positions.shape)
    print("Alpha position=",Alpha_pos);
    print("Beta position=",Beta_pos);
    print("Delta position=",Delta_pos);
    return Alpha_pos,Beta_pos;


# iters=100
# wolves=5
# dimension=13
# search_domain=[0,1]
# lb=-1.28
# ub=1.28
# colneeded=[0,1,2,4,5,7,8,10,11]
# modified_data=pd.DataFrame()
# for i in colneeded:
#     modified_data[data.columns[i]]=data[data.columns[i]].astype(float)
# func_details=benchmarks.getFunctionDetails(6)

# for i in range(0,10):
#     alpha,beta=binary_GWO_da(getattr(benchmarks,'F7'),lb,ub,dimension,wolves,iters)
    
##Applying feature selection on the given dataset
# ##considering alpha as best solution and putting a threshold
# threshold=-0.05
# index=[]
# print("alpha shape=",alpha.shape[0])
# modified_daata=pd.DataFrame();
# for i in range(0,alpha.shape[0]):
#     if(alpha[i]>=threshold):
#         modified_daata[data.columns[i]]=data[data.columns[i]].astype(float)
# print("The modified data is following")
# modified_daata.head()