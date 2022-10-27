import pandas as pd
from datetime import datetime
import os
import sys
import shutil
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


if __name__ == "__main__":
    #run = input('digite a url do run: ')
    run = sys.argv[1]
    run_splits = run.split('/')
    run_id = run_splits[-1]
    run_path = './mlruns/'+run_splits[5]+'/'+run_splits[-1]
    country = open(run_path+'/tags/data','r').read()
    size = open(run_path+'/tags/size','r').read()
    predictor = open(run_path+'/tags/model','r').read()
    true = pd.read_csv(run_path+'/artifacts/test_data.csv', index_col=0, parse_dates=True)
    preds = pd.read_csv(run_path+'/artifacts/preds.csv', index_col=0)
    preds_selecion = pd.read_csv(run_path+'/artifacts/'+predictor+'_test.csv', index_col=0)
    selection = pd.read_csv(run_path+'/artifacts/selection.csv', index_col=0, parse_dates=True)
    best = pd.read_csv(run_path+'/artifacts/best.csv', index_col=0, parse_dates=True)
    oracle = pd.read_csv(run_path+'/artifacts/oracle.csv', index_col=0, parse_dates=True)

    if not os.path.exists("charts/"+country+"/AEDS"):
            os.makedirs("charts/"+country+"/AEDS")
    template = ''
    for pred in preds.groupby('window'):
        #print(pred[0])
        print(pred[1].shape)
        fig, axs = plt.subplots(1, pred[1].shape[0], figsize=(20, 5))
        for model in pred[1].iterrows():
            k = model[0]
            model = model[1].drop('window')
            #print(model)
            #print(model.values)
            #print('model', model.values)
            #print('true', true.iloc[int(pred[0]):int(pred[0])+int(size)].values)
            #print(k)
            #print(axs)
            axs[k].plot(model, label='pred')
            axs[k].plot(true.iloc[int(pred[0]):int(pred[0])+int(size)].reset_index(drop=True), label='true')
            #print(model.shape)
            #print(true.iloc[int(pred[0]):int(pred[0])+int(size)].values.shape)
            mse = mean_squared_error(true.iloc[int(pred[0]):int(pred[0])+int(size)], model)
            #print(pred[0], k, mse)
            axs[k].annotate('MSE: '+ '{:,.0f}'.format(mse), xy=(0, -13),  xycoords='axes points', xytext=(0,-13), textcoords='offset points')
            axs[k].set_title('AE_'+str(pred[0])+'_'+str(k))
            axs[k].legend()
        plt.savefig("charts/"+country+"/AEDS/prediction"+str(pred[0])+".png")
        img = base64.standard_b64encode(open("charts/"+country+"/AEDS/prediction"+str(pred[0])+".png",'rb').read())

        try:
            template_base = '<div class="card"> \
            <div class="card-body"> \
                <div class="list-group"> \
                    <div class="list-group-item">week: {} </div> \
                    <img src="data:image/gif;base64,{}" /> \
                    <div class="list-group-item">selection: {}  oracle: {}</div> \
                    <div class="list-group-item">pred selection: {:,.0f}  pred oracle: {:,.0f}</div> \
                </div> \
            </div> \
        </div>'.format(pred[0], img.decode(),selection.iloc[pred[0]].values[0], best.iloc[int(pred[0])-1+int(size)].values[0], preds_selecion.iloc[pred[0]].values[0], oracle.iloc[int(pred[0])-1+int(size)].values[0])
            template = template + template_base
        except:
            pass
        
    """
    preds = pd.read_csv(run_path+'/artifacts/train_data.csv', index_col=0, parse_dates=True)
    val = pd.read_csv(run_path+'/artifacts/val_data.csv', index_col=0, parse_dates=True)
    train_meta = train.index[0].strftime('%d/%m/%y')+' - '+ train.index[-1].strftime('%d/%m/%Y') +', '+ str(train.shape[0])+ ' dias'
    val_meta = val.index[0].strftime('%d/%m/%Y') +' - '+ val.index[-1].strftime('%d/%m/%Y') +', '+ str(val.shape[0])+ ' dias'
    test_meta = test.index[0].strftime('%d/%m/%Y') +' - '+ test.index[-1].strftime('%d/%m/%Y') +', '+ str(test.shape[0])+ ' dias'
    img = base64.standard_b64encode(open(run_path+'/artifacts/data.png','rb').read())
    img2 = base64.standard_b64encode(open(run_path+'/artifacts/prediction.png','rb').read())
    ds_models = ['ASDS', 'ASO', 'ASVDS', 'ASVO', 'AEDS', 'AEO']
    report_type = 0
    
    ds_models.index(model)
    img3 = base64.standard_b64encode(open(run_path+'/artifacts/drifts.png','rb').read())
    drifts = open(run_path+'/tags/drifts','r').read()
    drifts_data = pd.read_csv(run_path+'/artifacts/drifts.csv', index_col=0, parse_dates=True)
    drifts_data[country] = drifts_data[country].diff(1)
    drifts_tab = drifts_data.dropna().groupby('drift').describe().applymap(lambda x: '{:,.2f}'.format(x)).to_html(classes='table table-striped')
    #drifts_tab = drifts_data.diff(1).dropna().describe().to_html(classes='table table-striped')
    selection = pd.read_csv(run_path+'/artifacts/best.csv', index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(selection, color='orange', label='selection')
    plt.legend()
    plt.savefig(run_path+'/artifacts/selection.png') 
    img4 = base64.standard_b64encode(open(run_path+'/artifacts/selection.png','rb').read())
    report_type = 1
    meta = {item.split(':')[0]: item.split(':')[1].strip(' ').strip('\n') for item in open(run_path+'/meta.yaml','r').readlines()}
    params = {m: open(run_path+'/params/'+m,'r').read() for m in os.listdir(run_path+'/params')}
    params = pd.DataFrame({'param':params.keys(), 'values':params.values()}).set_index('param').to_html(classes='table table-striped')
    metrics = {m: '{:,.2f}'.format(float(open(run_path+'/metrics/'+m,'r').read().split(' ')[1])) for m in os.listdir(run_path+'/metrics')}
    metrics = pd.DataFrame({'metric':metrics.keys(), 'values':metrics.values()}).set_index('metric').to_html(classes='table table-striped')
    
    #train_input = pd.read_csv(run_path+'/artifacts/input_train.csv', index_col=0)
    #train_output = pd.read_csv(run_path+'/artifacts/output_train.csv', index_col=0)
    #test_input = pd.read_csv(run_path+'/artifacts/input_test.csv', index_col=0)
    #train_output = pd.read_csv(run_path+'/artifacts/output_test.csv', index_col=0)
    print(model)
    print(report_type)

    if report_type == 1:
        template = open('reportds.html','r').read()
        report = template.format(
        datetime.fromtimestamp(int(meta['end_time'])/1000.0).strftime('%d/%m/%y %H:%M:%S'), 
        country, 
        train_meta, 
        val_meta, 
        test_meta, 
        img.decode(),
        drifts,
        drifts_tab,
        img3.decode(),
        model,
        params,
        metrics,
        img4.decode(),
        img2.decode()
        )
    else:
        template = open('report.html','r').read()
        report = template.format(
        datetime.fromtimestamp(int(meta['end_time'])/1000.0).strftime('%d/%m/%y %H:%M:%S'), 
        country, 
        train_meta, 
        val_meta, 
        test_meta, 
        img.decode(),
        model,
        params,
        metrics,
        img2.decode()
        )

    """
    if not os.path.exists("./reports/selection/"):
        os.makedirs("./reports/selection/")
    report = open('report_selection.html','r').read()
    report = report.format(template)
    #with open(run_path+'/tags/artifacts/'+run_id+'.html', 'w', encoding="utf-8") as f:
    with open('./reports/selection/'+run_id+'.html', 'w', encoding="utf-8") as f:
        read_data = f.write(report)
    f.close()
    
    shutil.copy('./reports/selection/'+run_id+'.html', run_path+'/artifacts/report.html')