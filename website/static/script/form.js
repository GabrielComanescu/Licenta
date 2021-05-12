let d2 = document.getElementById('col2');
let d1 = document.getElementById('col1');
let r = document.getElementById('right-arrow');
let s = document.getElementById('left-arrow');
let cent = document.getElementsByClassName('center');

function leftFunction(){
    
    d2.style.width = '0%';
    d1.style.width = '100%';

    s.style.display='block';
    r.style.display='none';

    let input_form = document.getElementById('upload');
    input_form.style.display='block'; 

    cent[0].style.top ='25%';
}

function rightFunction(){

    d2.style.width = '100%';
    d1.style.width = '0%';

    s.style.display='none';
    r.style.display='block';
}

function back_right(){
    d2.style.width = '50%';
    d1.style.width = '50%';

    s.style.display='none';
    r.style.display='none';
}


function back_left(){
    d2.style.width = '50%';
    d1.style.width = '50%';

    s.style.display='none';
    r.style.display='none';

    let input_form = document.getElementById('upload');
    input_form.style.display='none'

    cent[0].style.top ='50%';
}

let up = document.getElementById('upload');
up.onsubmit = async (e) => {
    e.preventDefault();

    let response = await fetch('http://127.0.0.1:5000', {
        method: 'POST',
        body: new FormData(up)
    })
    .then(resp => resp.blob()) 
    .then(function(data){

        let u = URL.createObjectURL(data);
        const link = document.createElement('a')
        link.href = u;
        link.download = 'upscaled';


        link.dispatchEvent(
            new MouseEvent('click', {
                bubbles:true,
                cancelable: true,
                view: window
            })
        );     
    })

    
}


