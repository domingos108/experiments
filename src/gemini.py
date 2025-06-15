import numpy as np

class SC_III_Algorithm:
    def __init__(self, L_max, epsilon, T_max, gamma_min, gamma_max, num_gamma_steps, lambda_limit=1.0):
        """
        Inicializa o algoritmo SC-III.

        Args:
            L_max (int): Número máximo de nós ocultos (neurônios) ou camadas.
            epsilon (float): Tolerância de erro esperada (para a norma de Frobenius do erro).
            T_max (int): Número máximo de tentativas de configuração de neurônios aleatórios em cada etapa.
            gamma_min (float): Valor mínimo para o escalar gamma.
            gamma_max (float): Valor máximo para o escalar gamma.
            num_gamma_steps (int): Número de passos para gerar o conjunto de gamma.
            lambda_limit (float): Limite para a atribuição aleatória de pesos e vieses [-\lambda, \lambda].
        """
        self.L_max = L_max
        self.epsilon = epsilon
        self.T_max = T_max
        # Gera o conjunto de gamma como um array NumPy
        self.gamma_set = np.linspace(gamma_min, gamma_max, num_gamma_steps)
        self.lambda_limit = lambda_limit

        # Listas para armazenar os pesos e vieses dos neurônios selecionados
        self.omega_star = [] # Pesos ótimos (omega*)
        self.b_star = []     # Vieses ótimos (b*)
        self.beta_star = []  # Pesos de saída ótimos (beta*)

    def _sigmoid(self, x):
        """Função de ativação sigmoide."""
        return 1 / (1 + np.exp(-x))

    def _calculate_h(self, X, omega, b):
        """
        Calcula a saída de um único neurônio oculto (h_L(X)).
        Assumimos que omega é um vetor coluna e X é uma matriz de entrada.
        """
        # (X @ omega) realiza o produto escalar de cada linha de X com omega
        # O resultado é um vetor de saídas para cada amostra
        return self._sigmoid(X @ omega + b)

    def _calculate_xi(self, e_L_minus_1_q, h_L_X):
        """
        Calcula a medida de importância xi_L,q conforme a equação fornecida.
        xi_L,q = (e_L-1,q^T(X) . h_L(X))^2 / (h_L^T(X) . h_L(X))

        Args:
            e_L_minus_1_q (np.ndarray): Vetor de erro/resíduo da camada anterior (e_L-1,q(X)).
                                        Deve ser 1D ou (N, 1).
            h_L_X (np.ndarray): Vetor de ativações do neurônio da camada atual (h_L(X)).
                                Deve ser 1D ou (N, 1).
        Returns:
            float: O valor de xi_L,q.
        """
        # Garante que ambos são vetores 1D ou 2D (coluna) para np.dot
        e_L_minus_1_q = e_L_minus_1_q.flatten()
        h_L_X = h_L_X.flatten()

        numerator = (np.dot(e_L_minus_1_q, h_L_X))**2
        denominator = np.dot(h_L_X, h_L_X)

        # Evita divisão por zero se h_L_X for um vetor nulo
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def fit(self, X, T):
        """
        Executa o Algoritmo 3 SC-III para treinar o modelo.

        Args:
            X (np.ndarray): Matriz de dados de entrada (N_samples, N_features).
            T (np.ndarray): Matriz de saídas alvo (N_samples, N_outputs).
        """
        N_samples, N_features = X.shape
        N_outputs = T.shape[1] if T.ndim > 1 else 1 # Lida com T sendo 1D ou 2D

        # 1. Initialize e0 := [t1, t2, ..., tN]^T, 0 < r < 1, Omega := [], W := []
        e0 = T.copy() # O erro inicial é o próprio target
        L = 0         # L (número de neurônios ocultos selecionados até agora)
        # r não é diretamente usado no fluxo do SC-III, mas é uma constante do SC-I
        # Ω e W serão gerenciados dentro da fase 1 do SC-I
        Omega_best_neurons = [] # Armazena (omega*, b*, xi_L) para cada neurônio selecionado
        H_L_matrix = np.empty((N_samples, 0)) # Matriz H_L (acumula as ativações h*)

        print("Iniciando o treinamento SC-III...")

        # 2. While L <= L_max AND ||e0||F > epsilon, Do
        # np.linalg.norm(e0, 'fro') calcula a norma de Frobenius (ou L2 para vetores)
        while L < self.L_max and np.linalg.norm(e0, 'fro') > self.epsilon:
            L += 1 # Incrementa L para a nova camada/neurônio

            print(f"\nIteração L = {L}/{self.L_max}. Erro atual ||e0||F = {np.linalg.norm(e0, 'fro'):.4f}")

            # 3. Proceed Phase 1 of Algorithm SC-I;
            # Esta fase busca o melhor neurônio (omega*, b*) para a camada atual.
            # O Algoritmo 1 SC-I parece procurar um neurônio por vez, que maximiza xi_L.
            max_xi_L = -np.inf
            best_omega_L = None
            best_b_L = None
            best_h_L_X = None # Armazena a ativação do melhor neurônio

            # Loop para T_max tentativas de configuração aleatória (Passos 4-10 do SC-I)
            for _ in range(self.T_max):
                # 5. Randomly assign omega_L and b_L
                # omega_L (N_features, 1) ou (N_features,)
                omega_L = np.random.uniform(-self.lambda_limit, self.lambda_limit, size=N_features)
                # b_L (1,)
                b_L = np.random.uniform(-self.lambda_limit, self.lambda_limit, size=1)[0] # Pega o escalar

                # 6. Calculate h_L(X)
                h_L_X = self._calculate_h(X, omega_L, b_L)

                # e_L_minus_1,q é e0 neste ponto (o erro residual atual)
                # Note: O algoritmo 1 usa e_L-1,q, o que implica que ele pode testar
                # contra diferentes colunas do erro se T for uma matriz.
                # Para simplificar e seguir a linha do Algoritmo 3 que usa T diretamente,
                # assumiremos que estamos calculando a contribuição de h_L(X) para CADA COLUNA de e0.
                # E então, o min(xi_L,q) é tomado sobre as colunas do erro.
                
                # Se T (e, portanto, e0) tiver múltiplas colunas (saídas),
                # calculamos xi para cada coluna e pegamos o mínimo.
                xi_values_for_current_neuron = []
                if e0.ndim > 1: # Múltiplas saídas
                    for q_col in range(e0.shape[1]):
                        xi_val = self._calculate_xi(e0[:, q_col], h_L_X)
                        xi_values_for_current_neuron.append(xi_val)
                    # Passos 7-8 do SC-I: If min(xi_L,1, ..., xi_L,m) >= 0
                    current_xi_L_min = np.min(xi_values_for_current_neuron)
                else: # Apenas uma saída (e0 é 1D)
                    current_xi_L_min = self._calculate_xi(e0, h_L_X)


                # 7. If min(xi_L,1, ..., xi_L,m) >= 0 (se todas as projeções são não-negativas)
                # 8. Save w_L and b_L, xi_L = sum(xi_L,q) in Omega
                # O algoritmo 1 salva SOMENTE se min(xi) >= 0.
                # No entanto, para otimização, geralmente queremos o que *maximiza* xi_L.
                # O passo 13 do Algoritmo 1 diz "Find w_L*, b_L* that maximize xi_L in Omega".
                # Vou usar a soma dos xi_q para representar xi_L para comparação.
                
                # Para este exemplo, vamos simplificar e maximizar a soma dos xi_q
                # ou o próprio xi_L se for um escalar
                current_xi_L_sum = np.sum(xi_values_for_current_neuron) if e0.ndim > 1 else current_xi_L_min

                if current_xi_L_sum > max_xi_L: # Encontrou um neurônio melhor
                    max_xi_L = current_xi_L_sum
                    best_omega_L = omega_L
                    best_b_L = b_L
                    best_h_L_X = h_L_X

            # Após as T_max tentativas, temos o melhor neurônio para esta camada
            if best_omega_L is None:
                # Se nenhum neurônio foi encontrado com min(xi) >= 0 no SC-I,
                # ou se simplesmente não houver um bom candidato.
                # O Algoritmo 1 tem um 'Else randomly take tau, renew r, return to Step 4'
                # Aqui, para o SC-III, se não encontramos nenhum, vamos parar ou dar um aviso.
                print("Aviso: Não foi possível encontrar um neurônio que maximize xi_L.")
                break # Sai do loop principal se nenhum neurônio foi encontrado
            
            # Adiciona o melhor neurônio e sua ativação à lista de neurônios selecionados
            self.omega_star.append(best_omega_L)
            self.b_star.append(best_b_L)
            
            # 4. Obtain H_L = [h_1*, h_2*, ..., h_L*];
            # Adiciona a ativação do neurônio recém-selecionado à matriz H_L
            H_L_matrix = np.hstack((H_L_matrix, best_h_L_X.reshape(-1, 1)))


            # 5. Calculate beta* = [beta_1*, ..., beta_L*]^T := H_L+ T;
            # np.linalg.pinv(H_L_matrix) calcula a pseudo-inversa de H_L
            # A multiplicação de matrizes H_L_matrix.T @ T resolveria para T sendo 1D
            # Mas H_L_matrix+ @ T é a solução de mínimos quadrados para a regressão
            
            # Garante que T tem 2 dimensões para a multiplicação de matrizes
            T_reshaped = T.reshape(N_samples, -1) if T.ndim == 1 else T
            
            beta_current = np.linalg.pinv(H_L_matrix) @ T_reshaped
            
            # Armazena os pesos de saída calculados para a camada atual L
            # Estes são os pesos que conectam os neurônios em H_L_matrix à saída T
            self.beta_star = beta_current.flatten().tolist() # Flatten para lista simples se for 1D


            # 6. Calculate e_L = H_L beta* - T;
            # Calcula o erro residual atual
            eL = (H_L_matrix @ beta_current) - T_reshaped

            # 7. Renew e0 := eL; L = L + 1; (L já foi incrementado no início do loop)
            e0 = eL.copy() # Atualiza o erro para a próxima iteração


        print(f"\nTreinamento concluído. Número final de neurônios ocultos (L): {L}")
        print(f"Norma de Frobenius final do erro: {np.linalg.norm(e0, 'fro'):.4f}")

    def predict(self, X_test):
        """
        Faz previsões usando os neurônios e pesos de saída aprendidos.

        Args:
            X_test (np.ndarray): Novos dados de entrada para fazer previsões.

        Returns:
            np.ndarray: As previsões do modelo.
        """
        if not self.omega_star or not self.b_star or not self.beta_star:
            raise ValueError("O modelo precisa ser treinado antes de fazer previsões.")

        N_test_samples, _ = X_test.shape
        
        # Constrói a matriz H para os dados de teste usando os neurônios selecionados
        H_test = np.empty((N_test_samples, len(self.omega_star)))
        for i, (omega, b) in enumerate(zip(self.omega_star, self.b_star)):
            H_test[:, i] = self._calculate_h(X_test, omega, b)
        
        # Multiplica as ativações pela pseudo-inversa calculada
        # (Neste algoritmo, beta_star já é o resultado final da pseudo-inversa)
        # Se beta_star é um vetor coluna, reshape para (num_neurons, num_outputs)
        beta_reshaped = np.array(self.beta_star).reshape(len(self.omega_star), -1)

        predictions = H_test @ beta_reshaped
        return predictions

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Gerar dados de exemplo: uma função não linear com ruído
    np.random.seed(42) # Para reprodutibilidade
    num_samples = 100
    num_features = 1
    
    X_train = np.linspace(-3, 3, num_samples).reshape(-1, num_features)
    # Exemplo de função alvo: y = x^2 + 2x - 1 + ruído
    Y_train = (X_train**2 + 2 * X_train - 1 + np.random.normal(0, 0.5, X_train.shape)).reshape(-1, 1)

    # Definir os parâmetros do algoritmo
    max_hidden_neurons = 20 # L_max
    error_tolerance = 0.5 # epsilon
    max_random_configs = 100 # T_max
    gamma_min_val = 0.1 # gama_min
    gamma_max_val = 1.0 # gama_max
    num_gamma_steps_val = 10 # num_gamma_steps
    lambda_val = 1.0 # lambda_limit para pesos e vieses

    # Criar e treinar o modelo
    sc_iii_model = SC_III_Algorithm(
        L_max=max_hidden_neurons,
        epsilon=error_tolerance,
        T_max=max_random_configs,
        gamma_min=gamma_min_val,
        gamma_max=gamma_max_val,
        num_gamma_steps=num_gamma_steps_val,
        lambda_limit=lambda_val
    )

    sc_iii_model.fit(X_train, Y_train)

    # Fazer previsões em novos dados
    X_test = np.linspace(-4, 4, 200).reshape(-1, num_features)
    Y_pred = sc_iii_model.predict(X_test)

    # Plotar os resultados
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, Y_train, label='Dados de Treino (Com Ruído)', alpha=0.7)
    plt.plot(X_test, X_test**2 + 2 * X_test - 1, color='red', linestyle='--', label='Função Verdadeira ($x^2 + 2x - 1$)')
    plt.plot(X_test, Y_pred, color='blue', label='Previsão do Modelo SC-III')
    plt.title('Aproximação de Função com Algoritmo SC-III')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nNúmero de neurônios ocultos selecionados: {len(sc_iii_model.omega_star)}")
    # print(f"Pesos dos neurônios (omega_star): {sc_iii_model.omega_star}")
    # print(f"Vieses dos neurônios (b_star): {sc_iii_model.b_star}")
    # print(f"Pesos de saída (beta_star): {sc_iii_model.beta_star}")