import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_handler
import pickle

# verifica se a senha de acesso está correta
if not util.check_password():
    # se a senha estiver errada, para o processamento do app
    st.stop()

dados = data_handler.load_data()

model = pickle.load(open('./models/model.pkl', 'rb'))

data_analyses_on = st.toggle("Exibir análise dos dados")

if data_analyses_on:
    st.dataframe(dados)

    st.header("Histograma das idades")
    fig = plt.figure()
    plt.hist(dados['Age'], bins=38)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

    # plota um gráfico de barras com a contagem dos sobreviventes
    st.header('Sobreviventes')
    st.bar_chart(dados.Survived.value_counts())

st.header("Preditor de sobrevivência")

col1, col2, col3 = st.columns(3)

with col1:
    classes = ["1st", "2nd", "3rd"]
    p_class = st.selectbox("Ticket class", classes)

with col2:
    classes = ["Male", "Female"]
    sex = st.selectbox('Genre', classes)

with col3:
    age = st.number_input("Age in years", step=1)

col1, col2, col3 = st.columns(3)

with col1:
    sib_sp = st.number_input("Number of siblings/spouses", step=1)

with col2:
    par_ch = st.number_input("Number of parents / children", step=1)

with col3:
    fare = st.number_input("Passanger Fare")

col1, col2 = st.columns(2)

with col1:
    classes = ["Cherbourg", "Queenstown", "Southampton"]
    embarked = st.selectbox("Port of Embarkation", classes)

with col2:
    submit = st.button("Verificar")

p_class_map = {
    '1st': 1,
    '2nd': 2,
    '3rd': 3
}

sex_map = {
    'Male': 0,
    'Female': 1
}

embarked_map = {
    'Cherbourg': 0,
    'Queenstown': 1,
    'Southampton': 2
}

passageiro = {
    "Pclass": p_class_map[p_class],
    "Sex": sex_map[sex],
    "Age": age,
    "SibSp": sib_sp,
    "Parch": par_ch,
    "Fare": fare,
    "Embarked": embarked_map[embarked]
}

values = pd.DataFrame([passageiro])
st.dataframe(values)

if submit or 'survived' in st.session_state:
    # TODO: verificar se o usuário informou todas as informações do passageiro antes de realizar o processamento dos dados e predição
    # TODO: no dataset possuiam vários dados sem a idade do passageiro, assim deveriamos permitir que a idade não fosse informada e tratar essa falta de informação

    # seta todos os attrs do passsageiro e já realiza o mapeamento dos attrs que não são numéricos
    passageiro = {
        'Pclass': p_class_map[p_class],
        'Sex': sex_map[sex],
        'Age': age,
        'SibSp': sib_sp,
        'Parch': par_ch,
        'Fare': fare,
        'Embarked': embarked_map[embarked]
    }
    print(passageiro)

    # converte o passageiro para um pandas dataframe
    # isso é feito para igualar ao tipo de dado que foi utilizado para treinar o modelo
    values = pd.DataFrame([passageiro])
    print(values)

    # realiza a predição de sobrevivência do passageiro com base nos dados inseridos pelo usuário
    results = model.predict(values)
    print(results)

    if len(results) == 1:
        # converte o valor retornado para inteiro
        survived = int(results[0])

        # verifica se o passageiro sobreviveu
        if survived == 1:
            # se sim, exibe uma mensagem que o passageiro sobreviveu
            st.subheader('Passageiro SOBREVIVEU! 😃🙌🏻')
            if 'survived' not in st.session_state:
                st.balloons()
        else:
            # se não, exibe uma mensagem que o passageiro não sobreviveu
            st.subheader('Passageiro NÃO sobreviveu! 😢')
            if 'survived' not in st.session_state:
                st.snow()

        # salva no cache da aplicação se o passageiro sobreviveu
        st.session_state['survived'] = survived


    if passageiro and 'survived' in st.session_state:
        st.write("A predicão está correta?")

        col1, col2, col3 = st.columns([1, 1, 5])

        with col1:
            correct_prediction = st.button('👍')

        with col2:
            wrong_prediction = st.button('👎')

        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback!"
            if wrong_prediction:
                message += " Iremos utiilizar estes dados para melhorar nossa predição."

            if correct_prediction:
                passageiro['CorrectPrediction'] = True
            elif wrong_prediction:
                passageiro['CorrectPrediction'] = False

            passageiro['Survived'] = st.session_state['survived']

            # escreve a mensagem na tela
            st.write(message)
            print(message)

            # salva a predição no JSON para cálculo das métricas de avaliação do sistema
            data_handler.save_prediction(passageiro)

    st.write('')
    # adiciona um botão para permitir o usuário realizar uma nova análise
    col1, col2, col3 = st.columns(3)
    with col2:
        new_test = st.button('Iniciar Nova Análise')

        # se o usuário pressionar no botão e já existe um passageiro, remove ele do cache
        if new_test and 'survived' in st.session_state:
            del st.session_state['survived']
            st.rerun()

# calcula e exibe as métricas de avaliação do modelo
# aqui, somente a acurária está sendo usada
# TODO: adicionar as mesmas métricas utilizadas na disciplina de treinamento e validação do modelo (recall, precision, F1-score)
accuracy_predictions_on = st.toggle('Exibir acurácia')

if accuracy_predictions_on:
    # pega todas as predições salvas no JSON
    predictions = data_handler.get_all_predictions()
    # salva o número total de predições realizadas
    num_total_predictions = len(predictions)

    # calcula o número de predições corretas e salva os resultados conforme as predições foram sendo realizadas
    accuracy_hist = [0]
    # salva o numero de predições corretas
    correct_predictions = 0
    # percorre cada uma das predições, salvando o total móvel e o número de predições corretas
    for index, passageiro in enumerate(predictions):
        total = index + 1
        if passageiro['CorrectPrediction'] == True:
            correct_predictions += 1

        # calcula a acurracia movel
        temp_accuracy = correct_predictions / total if total else 0
        # salva o valor na lista de historico de acuracias
        accuracy_hist.append(round(temp_accuracy, 2))

        # calcula a acuracia atual
    accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0

    # exibe a acuracia atual para o usuário
    st.metric(label='Acurácia', value=round(accuracy, 2))
    # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia

    # exibe o histórico da acurácia
    st.subheader("Histórico de acurácia")
    st.line_chart(accuracy_hist)