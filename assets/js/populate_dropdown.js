// Load all images in a specified directory and update the carousel HTML
let contexts={
    cat:"cat wearing sunglasses",
    panda:"panda riding a bike",
    sub:"submarine",
    eiffel:"image of eiffel tower",
    pug:"fat pug wearing a cowboy hat",
    mountain:"snow capped mountain range",
    ship:"ship at sea in a storm",
    city:"city skyline at sunset",
    minions:"Minions"
}

let styles={
    van_gogh:"in the style of Van Gogh",
    dali:"as a painting of Dalí (1929), surrealist",
    klimt:"in the style of Gustav Klimt",
    north_korea:"in the style of North Korean propaganda poster",
    bauhaus:"in the style of bauhaus art",
    pencil:"in the style of a pencil sketch,line drawing",
    picasso:"in the style of Picasso, Cezanne, Cubism",
    monet:"in the style of Monet",
    pollock:"in the style of Jackson Pollock",
    warhol:"in the style of Warhol, pop art",
    children:"as a children's drawing",
    mosaic:"as a Byzantine mosaic",
}

// Get the dropdown menu and repopulate with values, set id to fields
let context=document.getElementById('Context_dropdown');
let style=document.getElementById('Style_dropdown');

// Remove the current values and populate with list
context.innerHTML='';
for (const [key, value] of Object.entries(contexts)){
    context.innerHTML+="<option value="+key+">"+value+"</option>"
}
style.innerHTML='';
for (const [key,value] of Object.entries(styles)){
    style.innerHTML+="<option value="+key+">"+value+"</option>"
}

let image_folder=context[context.selectedIndex].value+"_"+style[style.selectedIndex].value

let image_grid=document.getElementById('ai_grid')

function populateImageGrid(image_grid,image_folder){
    image_grid.innerHTML='';
    image_path=document.location.origin+"/assets/img/ai_art/dropdown_select/"+image_folder+"/samples/enchanced/";
    for (let i=0; i<=15; i++){
        image_grid.innerHTML+='<div class="col"><img src='+image_path+("000"+i).slice(-4)+'_out.png '+'alt="image" class="img-fluid" /></div>'
    }
}

populateImageGrid(image_grid,image_folder);

context.addEventListener('change',function(){
    image_folder=context[context.selectedIndex].value+"_"+style[style.selectedIndex].value;
    populateImageGrid(image_grid,image_folder);
})

style.addEventListener('change',function(){
    image_folder=context[context.selectedIndex].value+"_"+style[style.selectedIndex].value;
    populateImageGrid(image_grid,image_folder);
})