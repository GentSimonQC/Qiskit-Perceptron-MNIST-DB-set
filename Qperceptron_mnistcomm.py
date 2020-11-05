
import Generative_Layer_for_DB_MNIST_FWD as GL
import DB_MNIST_checkpunctual


from math import *
import numpy as np
from qiskit import *
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit import IBMQ
import matplotlib.pyplot as plt
import operator


num_classes = 2
classesarray = list(range(num_classes)) # 0, 1... CLASSES APPENDED TO OVERALL[] IN qthresholds. Read it as 'IS 'X' ELEMENT FROM CLASS 0 OR FROM CLASS 1?'
MAX_DATA = 25   # 'BATCH' SIZE

DB_MNIST_checkpunctual.SIZE = 11 #11-14
DB_MNIST_checkpunctual.SIDEDIV = 4 #3-5
ENTROPY_THRESHOLD = 1.0


#===============================================================================
# PARAMETERS AND INIT
#===============================================================================
REAL_DEVICE = False
REVERSE = False


what_number_label = 3 # WHAT IMAGE IS GIVEN IN INPUT (THE ALGO DOES NOT KNOW WHAT NUMBER IT IS), GL.example_main(what_number_label=what_number_label

the_other_class_num_array = [4] #[3,4,7]  # ALL THE NUMBERS THE INPUT WILL BE COMPARED AGAINST

finalsols_maxnum = 2

step = 3        # !

divisorpower = 1.9    # >=2 gives more and more relevance to gamma_fun values [-1,1], < 2 gives relevance to its sign only    # !!!!!





def input_state(circ, n, theta):
    """special n-qubit input state"""
    for j in range(n):
        circ.h(j)
        circ.u1(theta*float(2**(j)), j)
        #

        
def qft(circ, n, theta):
    """special n-qubit QFT on the qubits in circ."""
    for j in range(n):
        for k in range(j+1,n):
            circ.cu1(theta, k, j)
            #
        circ.barrier()
    circ.cx(1, 2)
    circ.cu3(np.arccos(theta/6), theta, np.arccos(theta/6), 0, 2)
    swap_registers(circ, n)


def qft_rev(circ, n, theta):
    """special alternative n-qubit QFT on the qubits in circ."""
    for j in range(n):
        for k in range(j+1,n):
            circ.cu1(theta, k, j)
            #
        circ.barrier()
    circ.cx(1, 2)
    circ.cu3(exp(sin(theta)), pi*theta/3, exp(-theta), 0, 2)
    swap_registers(circ, n)
    circ.cx(0, 1)
    for jj in range(n):
        circ.h(jj)


def swap_registers(circ, n):
    for j in range(int(np.floor(n/2.))):
        circ.swap(j, n-j-1)
    return circ


def circuit_calcs(nqubits, XX, what_model):


    qft_circuit = QuantumCircuit(nqubits)
    
    # first, prepare the state
    gamma_fun = 0
    
    
    if what_model == 'straight':

        FIRST_COL = 0

        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += (XX[0][i+k]-XX[0][i])*cos(XX[0][k+1-i]-XX[0][i+2])/(np.shape(XX)[1]**divisorpower)
        
        print('gamma_fun start',gamma_fun)
    
        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1
    
        input_state(qft_circuit, nqubits, 1-np.cosh(gamma_fun))
        # qft_circuit.draw(output='mpl')
        # plt.show()
        
        # next, do our modified qft on the prepared state
        qft_circuit.barrier()
        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += tanh(XX[0][i+k]-XX[0][-1])/(np.shape(XX)[1]**divisorpower)

        print('gamma_fun end',gamma_fun)
    
        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1
        
    
        qft(qft_circuit, nqubits, -pi*gamma_fun)



    elif what_model == 'reverse':
    
    # first, prepare the state
        
        FIRST_COL = 4
        
        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += (XX[0][i+k]-XX[0][i])*exp(sin(XX[0][k+1-i]-XX[0][i+2]))/(np.shape(XX)[1]**divisorpower)

        print('gamma_fun start',gamma_fun)

        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1

        input_state(qft_circuit, nqubits, 1-np.cosh(gamma_fun))
        # qft_circuit.draw(output='mpl')
        # plt.show()
        
        # next, do our modified qft on the prepared state
        qft_circuit.barrier()
        for i in range(FIRST_COL, np.shape(XX)[1] - 2, step):
            for k in range(0, np.shape(XX)[1] - 1, step):
                if i+k >= np.shape(XX)[1]:
                    break
                gamma_fun += tanh(XX[0][i+k]-XX[0][-1])/(np.shape(XX)[1]**divisorpower)

        print('gamma_fun end',gamma_fun)

        if gamma_fun < -1:
            gamma_fun = -1
        elif gamma_fun > 1:
            gamma_fun = 1
           

        qft_rev(qft_circuit, nqubits, -pi*gamma_fun)
    
    
    qft_circuit.measure_all()
    
    
    return qft_circuit


def qcounts(qcircuit, nqubits, REAL_DEVICE):
    
    if REAL_DEVICE == True:
        # run on real Q device
        #=======================================================================
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= nqubits and
                                           not x.configuration().simulator and x.status().operational==True))
        print("least busy backend: ", backend)
        shots = 8192
        job_exp = execute(qcircuit, backend=backend, shots=shots)
        job_monitor(job_exp)
        print(job_exp.error_message())
        fourier_counts = job_exp.result().get_counts()
        #plot_histogram([fourier_counts], legend=['counts'])
        #plt.show()
    else:
        # run on local simulator
        #=======================================================================
        shots = 8192
        qasm_sim = Aer.get_backend("qasm_simulator")
        fourier_counts = execute(qcircuit, backend=qasm_sim, shots=shots).result().get_counts()
        #plot_histogram([fourier_counts], legend=['counts'])
        #plt.show()    

    return fourier_counts




def qthresholds(fourier_counts, reverse, what_model):

    OVERALL = []
    
    
    if what_model == 'straight':

    
        if '110' in fourier_counts and '111' in fourier_counts:
            
            if REVERSE == False:
    
                if fourier_counts['110'] < fourier_counts['111']:
                    OVERALL.append(classesarray[len(classesarray)-1])
                else:
                    OVERALL.append(classesarray[0])
                
            else:
                
                if fourier_counts['110'] < fourier_counts['111']:
                    OVERALL.append(classesarray[0])
                else:
                    OVERALL.append(classesarray[len(classesarray)-1])        

        else:
            OVERALL.append(classesarray[1])
    

    elif what_model == 'reverse':


        if '001' in fourier_counts and '101' in fourier_counts:
            
            if REVERSE == False:
    
                if fourier_counts['001'] < fourier_counts['101']:
                    OVERALL.append(classesarray[len(classesarray)-1])
                else:
                    OVERALL.append(classesarray[0])
            
            else:
                
                if fourier_counts['001'] < fourier_counts['101']:
                    OVERALL.append(classesarray[0])
                else:
                    OVERALL.append(classesarray[len(classesarray)-1])
    
        else:
            OVERALL.append(classesarray[1])



    return OVERALL





random_index = 12



MAX_SOLUTIONS = 9


SOLUTIONS = np.empty((MAX_SOLUTIONS,len(the_other_class_num_array)))
SOLUTIONS[0] = the_other_class_num_array
for k in range(1, np.shape(SOLUTIONS)[0]):
    SOLUTIONS[k] = -1
#print(SOLUTIONS)
def emptycheck(array):
    
    ok = False
    counter = 0
    for e in array:
        if e >= 0 and e < 500:
            ok = True
            counter += 1
        
    return (not ok), counter
            

cyclesolutions = 0  # EVERY cyclesolutions CYCLE SHOULD NARROW THE CLASSIFICATION UNCERTAINTY

while (cyclesolutions < np.shape(SOLUTIONS)[0] - 1):



    
    MODEL_TYPE = ['straight','reverse']
    ERRORSCORES = -1 * np.ones((len(MODEL_TYPE),len(the_other_class_num_array)))
    ENTROPY = 500 * np.ones((len(the_other_class_num_array)))
    
    
    
    forsolutionsarray = np.where(SOLUTIONS[cyclesolutions] >= 0)[0]
    forsolutionsarray = np.array(forsolutionsarray, dtype=int)
    print('forsolutionsarray',forsolutionsarray)
    # LOOP FOR THE ENTRIES forsolutionsarray
    for everyclass in forsolutionsarray:
    
    
        # DATABASE IMPORT
        FLATTEN_MERG_MATR = GL.example_main(what_number_label=what_number_label, tuple_shape=(DB_MNIST_checkpunctual.SIZE,DB_MNIST_checkpunctual.SIZE), max_data=MAX_DATA, the_other_class_num=the_other_class_num_array[everyclass])
        
        print('CHECK max min',np.max(FLATTEN_MERG_MATR),np.min(FLATTEN_MERG_MATR))
        
        X = np.exp(np.array(FLATTEN_MERG_MATR))   #FOR NORMALIZED [0, 1] DB

        
        Y = []
        CY = []
        for jj in range(MAX_DATA):
            Y.append(the_other_class_num_array[everyclass])
            CY.append(0)
        for jj in range(MAX_DATA):
            Y.append(the_other_class_num_array[everyclass])
            CY.append(len(classesarray)-1)
        
        print(Y,'shape(Y)',np.shape(Y))
        print(CY,'shape(CY)',np.shape(CY))
        
    
        TOTAL_SCANNED_RESULT  = -1 * np.ones((len(MODEL_TYPE), MAX_DATA*2))
        TOTAL_SCANNED_ERRORS_AS_CLASS = -1 * np.ones((len(MODEL_TYPE), MAX_DATA*2))
        TOTAL_UNKNOWNS = -1 * np.ones((len(MODEL_TYPE), MAX_DATA*2))
    
                
        start_for = 0
        end_for = np.shape(X)[0]
        
        for model in range(len(MODEL_TYPE)):    
        
            
            counter = 0
            for index in range(start_for,end_for):
    
                row = index   
                XX = [X[row]]
                OVERALL = []
    
                #===============================================================================
                # FOURIER T
                #===============================================================================
                nqubits = 3
                qft_circuit = circuit_calcs(nqubits=nqubits, XX=XX, what_model=MODEL_TYPE[model])
                
                fourier_counts = qcounts(qcircuit=qft_circuit, nqubits=nqubits, REAL_DEVICE=REAL_DEVICE)
        
        
                #key_list = list(fourier_counts.keys())
                #sorted_C = sorted(fourier_counts.items(), key=operator.itemgetter(1), reverse=True)
                #print(sorted_C[:2])
        
                # THRESHOLDS
                #=======================================================================
                OVERALL = qthresholds(fourier_counts=fourier_counts, reverse=REVERSE, what_model=MODEL_TYPE[model]) 
                        
                        
                
                print('end of cycle:',MODEL_TYPE[model],counter,'\n')
                counter += 1
            
                
                
                RESULT = 999
                print('OVERALL',OVERALL)
                binc = np.bincount(OVERALL) 
                print('binc',binc)
                if len(OVERALL) > 0:
                    RESULT = np.argmax(binc) # RESULT IS JUST THE CLASS VALUE (WITHIN classesarray VALUES) 
                    # AFTER qthresholds COMPARISONS BETWEEN OUR PIVOTAL COUNTS ('110' and '111', '001' and '101')
                
                #
                CRESULT = RESULT
                print('\n',row,'of check',what_number_label,'against num',the_other_class_num_array[everyclass],'cyclesolutions:',cyclesolutions,' max data:',MAX_DATA,'\n','\n')
            
            
                if CRESULT != 999: # FOR EXCLUDING ANY ERRORS IN ENCODING THE RESULT
    
                    print('RESULT CLASS:',CRESULT,'( WITH MODEL=',MODEL_TYPE[model],')','REAL CLASS TO GUESS:',CY[row])
                    TOTAL_SCANNED_RESULT[model,index] = CRESULT        
            
                    if abs(CRESULT - int(CY[row])) > 0:
                        TOTAL_SCANNED_ERRORS_AS_CLASS[model,index] = 1
                    else:
                        TOTAL_SCANNED_ERRORS_AS_CLASS[model,index] = 0

            
                else:
            
                    print('RESULT CLASS: UNDEFINED')
                    TOTAL_UNKNOWNS[model,index] = 1
            
    
    
            if len(TOTAL_SCANNED_RESULT[model]) > 0:
                print('MODEL',MODEL_TYPE[model],'TOTAL_SCANNED_ERRORS_AS_CLASS',TOTAL_SCANNED_ERRORS_AS_CLASS[model],'\n',
                      '!!! CLASS ERROR % !!!:',np.sum(TOTAL_SCANNED_ERRORS_AS_CLASS[model])/np.shape(TOTAL_SCANNED_ERRORS_AS_CLASS)[1]*100,'\n',
                      'TOTAL UNKNOWNS %',np.sum(TOTAL_UNKNOWNS[model])/np.shape(TOTAL_UNKNOWNS)[1]*100)
        
            ERRORSCORES[model,everyclass] = np.sum(TOTAL_SCANNED_ERRORS_AS_CLASS[model])/np.shape(TOTAL_SCANNED_ERRORS_AS_CLASS)[1]*100
    
    
    
        print('ERROR SCORES:',ERRORSCORES,'\n','ON THE HIDDEN NUMBER TO GUESS:',what_number_label,'against num',the_other_class_num_array[everyclass])
    
        for i in forsolutionsarray:
            sumentr = 0
            for m in range(len(MODEL_TYPE)):
                if ERRORSCORES[m,i] < 0:
                    continue
                sumentr += ( ERRORSCORES[m,i]/100 * log(1/(ERRORSCORES[m,i]/100)) )
            ENTROPY[i] = min(ERRORSCORES[:,i])/100 * np.exp(np.cosh(-sumentr))
        
        
        
    print('ENTROPY:',ENTROPY,'\n','ON THE HIDDEN NUMBER TO GUESS:',what_number_label)  # THE LOWER, THE MORE CONSISTENT
    
    cyclesolutions += 1


    if cyclesolutions >= 5:
        break


    for kk in range(len(ENTROPY)):
        if ENTROPY[kk] < 500 and ENTROPY[kk] > ENTROPY_THRESHOLD and ENTROPY[kk] < 2.05:
            SOLUTIONS[cyclesolutions][kk] = ENTROPY[kk]
            

    # CHECK DEEPER
    MAX_DATA += 5 #10


    CHECK_MIN = []
    # CONDITIONS FOR ENDING THE MAIN LOOP
    if emptycheck(SOLUTIONS[cyclesolutions])[1] <= finalsols_maxnum and cyclesolutions > 0:
        for ggg in range(np.shape(SOLUTIONS)[0]-1, 0, -1):
            if emptycheck(SOLUTIONS[ggg])[0] == True:
                continue
            else:

                if cyclesolutions > 0: #1:
                    CHECK_MIN.append(np.where(np.logical_and(ENTROPY > ENTROPY_THRESHOLD, ENTROPY < 2.05))[0])
                    # IF ONLY finalsols_maxnum CLASSES OF NUMBERS ARE LEFT, THEN CHOOSE THE ONE WITH THE HIGHEST FREQUENCY (=np.argmax(counts))
                    # OF NON-NULL ENTROPY VALUES (WITHIN THE RANGE [ENTROPY_THRESHOLD, 2.05], ACROSS ALL THE cyclesolutions
                    CHECK_MIN = np.array(CHECK_MIN).flatten()
                    counts = np.bincount(CHECK_MIN)
                    print('BINARY GUESS:',the_other_class_num_array[np.argmax(counts)])
                    break

                
                break
        
        break
     

    

print('SOLUTIONS:','\n',SOLUTIONS)