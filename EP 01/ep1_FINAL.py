"""
Tarefa 1 - Decomposição LU para Matrizes Tridiagonais - MAP3121
Data de entrega: 01/05/2022

    Rodrigo Gebara Reis - NUSP: 11819880
    Victor Rocha da Silva - NUSP: 11223782
"""


import numpy as np
import time


# Decomposição LU:
    
# Entradas: a matriz extendida A|b, tal que A array n x n, b array 1 x n (Ax = b)
def gauss(a):
    n = a.shape[0]

    a = np.array(a, dtype = float)
    A = np.copy(a[:,:-1])
    b = np.copy(a[:,n]).reshape((n))
    
    start = time.time()

    # Eliminação (Escalonamento)
    for i in range(n):
        for j in range(i+1, n):
            a[j,:] = a[j,:] - a[j, i]/a[i, i]*a[i,:]
    print(a)
    # Solução
    x = np.zeros(n)
    x[n-1] = a[n-1, n]/a[n-1, n-1]
    for i in range(n-2,-1,-1):
        x[i] = (a[i, n] - np.dot(a[i, i+1:n], x[i+1:n]))/a[i, i]

    end = time.time()

    print("Solucao: ")
    print(x)

    print("Residuo: ")
    print(np.max(np.abs(A@x - b)))

    print("Tempo: ")
    print(end - start)

    return x


# Decomposição LU (mais eficiente):

def decomposicaoLU(A): # A array n x n
    n = A.shape[0]
    L, U = np.zeros((n, n)), np.zeros((n, n))
    
    for i in range(n):
        U[i, i:n] = A[i, i:n] - L[i, :i]@U[:i, i:n]
        L[i+1:n, i] = (A[i+1:n, i] - L[i+1:n,:i]@U[:i, i])/U[i,i]
       
    np.fill_diagonal(L,1) # Completa a diagonal de L com 1

    return L, U

# # O array A (n x n) pode ser alterado para teste!
# A = np.array([[3,2,4],[1,1,2],[4,3,-2]])
# L, U = decomposicaoLU(A)

# print('\nL = ')
# print(L)
# print('\nU = ')
# print(U)
# print('\nA = ')
# print(L@U)

# Tarefa - Parte 1: decomposição LU de uma matriz tridiagonal A (n × n):
    
def LU_completo(A):
    
    n = A.shape[0]

    start = time.time()

    LU = np.eye(n) # Inicia matriz identidade de ordem n

    for i in range(n):
        #Varre linhas superiores (Upper)
        LU[i,i:] = A[i,i:]-LU[i,:i] @ LU[:i,i:]
        #Varre colunas inferiores (Lower)
        LU[(i+1):,i] = (A[(i+1):,i]- LU[(i+1):,:i] @ LU[:i,i]) / LU[i,i]

    U = np.triu(LU)
    L = np.tril(LU)
    np.fill_diagonal(L, 1)
    
    end = time.time()
    print("Tempo: ")
    print(end - start)

    print("L = ", L, "\n U = ", U, "\n LU = ", LU)
    return LU

# Função auxiliar para estruturar matrizes tridiagonais (cíclicas ou não)
def createMatrix(a, b, c):
  n = a.shape[0]

  A = np.zeros([n, n])
  A[0, n-1] = a[0]
  A[n-1, 0] = c[n-1]
  A[n-1, n-1] = b[n-1]
  for i in range(0, n-1):
    A[i, i] = b[i]
    A[i+1, i] = a[i+1]
    A[i, i+1] = c[i]

  return A


"""
Decomposição LU - Matriz tridiagonal:

Função 'decompLU' decompõe uma matriz A tridiagonal em uma matriz triangular 
inferior (L) e uma triangular superior (U). Como A é tridiagonal, é 
caracterizada por três vetores (a, b, c), fornecidos na entrada como arrays
numpy. A função retorna os arrays numpy (vetores) l e u que caracterizam as 
matrizes L e U.
"""

def decompLU(a, b, c):
  n = a.shape[0]
  l, u = np.zeros(n), np.zeros(n) # Inicia os vetores l e u

  l[0] = 0 # Assim como a[0] = 0, definimos l[0] = 0
  u[0] = b[0]
  
  # Segue as formulações definidas no texto acima
  for i in range(1, n):
    l[i] = a[i]/u[i-1]
    u[i] = b[i] - l[i]*c[i-1]
    
  return(l, u)


"""
Solução - Sistema Tridiagonal:

São aplicadas as formulações apresentadas no texto para resolver um sistema do 
tipo Ax = d, com A sendo uma matriz tridiagonal.
Na entrada são fornecidos os três vetores (a, b, c) que definem a matriz A, 
além do vetor d. Todos são arrays numpy. A função devolve o vetor solução x
"""

def resolveSistemaTridiagonal(a, b, c, d):
  n = a.shape[0]
  # Decompõe a matriz tridiagonal em uma triangular inferior, caracterizada pelo vetor l,
  # e uma triangular superior, caracterizada pelos vetores u e c
  l, u = decompLU(a, b, c)

  #Ly = d
  y = np.zeros(n)

  y[0] = d[0]
  for i in range(1, n):
    y[i] = d[i] - l[i]*y[i-1]

  #Ux = y
  x = np.zeros(n)

  x[n-1] = y[n-1]/u[n-1]
  for i in range(n-2, -1, -1):
    x[i] = (y[i]-c[i]*x[i+1])/u[i]

  return x


# # Os vetores 'a','b','c' e 'd' podem ser alterados para teste!
# a = np.array([0, 3., 1, 3])
# b = np.array([10., 10., 7., 5.])
# c = np.array([2., 4., 5., 0])
# d = np.array([3, 4, 5, 6.])

# start = time.time()
# x = resolveSistemaTridiagonal(a, b, c, d)
# end = time.time()

# print("\nSolucao: ")
# print(x)

# A = createMatrix(a, b, c)

# print("\nResiduo: ")
# print(np.max(np.abs(A@x-d)))

# print("\nTempo: ")
# print(end - start)


"""
# Tarefa - Parte 2: algoritmo para a resolução de sistema linear tridiagonal: 
(usando a decomposição LU da matriz)

Aplica as formulações apresentadas no texto para resolver um sistema tridiagonal
cíclico Ax = d. Como A é uma matriz tridiagonal, são fornecidos na entrada os 
três vetores que a definem (a, b, c), e o vetor d. A função devolve o vetor 
solução x.
"""

def resolveSistemaCiclico(a, b, c, d):
  n = a.shape[0]

  # Montar matriz T
  at = a[0:n-1].copy()
  at[0] = 0
  bt = b[0:n-1].copy()
  ct = c[0:n-1].copy()
  ct[-1] = 0

  # Montar vetor d~
  dt = d[0:n-1].copy()

  # Montar vetor v
  v = np.zeros(n-1)
  v[0] = a[0]
  v[n-2] = c[n-2]

  # Montar vetor w
  w = np.zeros(n)
  w[0] = c[n-1]
  w[-1] = a[n-1]

  # Solução sistema Ty = d~
  y = resolveSistemaTridiagonal(at, bt, ct, dt)

  # Solução sistema Tz = v
  z = resolveSistemaTridiagonal(at, bt, ct, v)

  x_n = (d[n-1]-c[n-1]*y[0]-a[n-1]*y[n-2])/(b[n-1]-c[n-1]*z[0]-a[n-1]*z[n-2])

  x = y - x_n*z

  x = np.append(x, x_n)

  return x

## Teste:

# Cria os vetores sugeridos para o teste que definem a matriz tridiagonal cíclica a ser resolvida no sistema
# n = tamanho da matriz utilizada para teste
def criaMatrizTeste(n):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(1, n):
        a[i-1] = (2*i-1)/(4*i)
        c[i-1] = 1-a[i-1]
        b[i-1] = 2
        d[i-1] = np.cos((2*(np.pi)*(i**2))/(n**2))
    
    a[n-1] = (2*n-1)/(2*n)
    c[n-1] = 1-a[n-1]
    b[n-1] = 2
    d[n-1] = 1
    
    return a,b,c,d

# n = tamanho da matriz utilizada para teste
def teste(n):
  a, b, c, d = criaMatrizTeste(n) 

  start = time.time()

  x = resolveSistemaCiclico(a, b, c, d)

  end = time.time()

  print('n =',n)  

  print("\nSolucão: ")
  print(x)

  A = createMatrix(a, b, c)
  print("\nResíduo: ")
  print(np.max(np.abs(A@x - d)))

  print("\nTempo: ")
  print(end-start)

  return

# # Teste para n=20 (sugestão da tarefa):
# teste(20)

# # Teste usando n=10000:
# # (apenas para mostrar que o tempo de execução está ok e que o resíduo continua
# # pequeno mesmo para matrizes grandes)
# teste(10000)


if __name__ == '__main__':
  # chamar as funções aqui para realizar os testes
  print('Chame a função que deseja testar ao final do código!')
