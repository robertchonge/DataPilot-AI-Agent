import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF


DATA = {
    "raw": None,
    "clean": None
}


def clean_dataset(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ","_")
    )


    df = df.drop_duplicates()


    for col in df.select_dtypes(
        include=np.number
    ):

        df[col] = df[col].fillna(
            df[col].median()
        )


    for col in df.select_dtypes(
        include="object"
    ):

        df[col] = df[col].fillna(
            "Unknown"
        )


    return df



def upload_data(file):

    if file.name.endswith(".csv"):

        df = pd.read_csv(
            file.name
        )

    else:

        df = pd.read_excel(
            file.name
        )


    DATA["raw"] = df

    DATA["clean"] = clean_dataset(df)


    return f"""
Dataset loaded successfully

Rows:
{df.shape[0]}

Columns:
{df.shape[1]}
"""



def preview():

    return DATA["clean"].head(20)



def dashboard():

    df = DATA["clean"]


    region = (
        df.groupby("region")
        ["total_sales"]
        .sum()
        .reset_index()
    )


    fig1 = px.bar(
        region,
        x="region",
        y="total_sales",
        title="Sales By Region"
    )


    product = (
        df.groupby("product_name")
        ["total_sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )


    fig2 = px.bar(
        product,
        x="total_sales",
        y="product_name",
        orientation="h",
        title="Top Products"
    )


    return fig1,fig2



def analyst(question):

    df = DATA["clean"]


    q = question.lower()


    if "profit" in q:

        return (
            f"Total profit is "
            f"{df['net_profit'].sum():,.2f}"
        )


    if "region" in q:

        result = (
            df.groupby("region")
            ["total_sales"]
            .sum()
            .idxmax()
        )


        return (
            f"Highest sales region: {result}"
        )


    if "product" in q:

        result = (
            df.groupby("product_name")
            ["total_sales"]
            .sum()
            .idxmax()
        )


        return (
            f"Best product: {result}"
        )


    return (
        "Ask about sales, profit, regions or products."
    )



with gr.Blocks(
    title="DataPilot AI"
) as demo:


    gr.Markdown(
    """
    # 🤖 DataPilot AI

    Your Intelligent Data Scientist
    """
    )


    with gr.Tab("Upload"):

        file = gr.File()

        status = gr.Textbox()


        file.upload(
            upload_data,
            file,
            status
        )



    with gr.Tab("Preview"):

        button = gr.Button(
            "Show Data"
        )

        table = gr.Dataframe()


        button.click(
            preview,
            outputs=table
        )



    with gr.Tab("Dashboard"):

        button = gr.Button(
            "Generate Dashboard"
        )

        chart1 = gr.Plot()

        chart2 = gr.Plot()


        button.click(
            dashboard,
            outputs=[
                chart1,
                chart2
            ]
        )



    with gr.Tab("AI Analyst"):

        question = gr.Textbox()

        answer = gr.Textbox()


        ask = gr.Button(
            "Ask DataPilot"
        )


        ask.click(
            analyst,
            question,
            answer
        )



demo.launch()
