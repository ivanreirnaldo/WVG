function [resp]=WVG_cone(nvar,npop,n,vector,tau)
% Cone of Weight Vector Generator version 1.0
% nvar   -> dimension of the objective space
% npop   -> number of weight vecrtors
% n      -> p-norm of the vectors
% vector -> axis of the cone
% tau    -> generatrix angle

% Algorithm settings
itermax=6000;           % max iterations
neighbors=nvar;         % number of neighbors
gap=0;
lb=zeros(1,nvar) + gap; % lower bound
ub=ones(1,nvar) - gap;  % upper bound
a=1.0;
b=0.1;

% Vector adressing
variables=1:nvar;
obj=nvar+1;
angle=obj+1;

% Population start
pop=zeros(npop,angle);
pop(:,variables)=gerapop(nvar,npop,lb,ub);

% vector normalizarion
for i=1:npop
    pop(i,variables)=pop(i,variables)/norm(pop(i,variables),n);
    pop(i,angle)=acos((pop(i,variables)*vector')/norm(pop(i,variables)*norm(vector)));
    if pop(i,angle)<tau
        pop(i,angle)=0;        
    end
end

dist=inf(npop,npop);
for i=1:npop
    for j=(i+1):npop
        dist(i,j)=norm(pop(i,variables)-pop(j,variables));
        dist(j,i)=dist(i,j);
    end
end
sigma=min(dist);
for i=1:npop
    pop(i,obj)=fo(dist(:,i),neighbors) - (nvar)*pop(i,angle);
end

% start algorithm
ok=0;
iter=1;
t=iter/itermax;
w=(1-t)*a + t*b;
sigma=w*sigma;

while ok==0
    y=zeros(1,angle);
    dist_y=zeros(1,npop);
    i=randi(npop,1);       
        delta=normrnd(0,w*sigma(i),1,nvar);
        y(variables)=pop(i,variables) + delta;
        y(variables)=max(y(variables),lb);
        y(variables)=min(y(variables),ub);
        y(variables)=y(variables)/norm(y(variables),n);
        y(angle)=acos((y(variables)*vector')/(norm(y(variables)*norm(vector))));
        if y(angle)<tau
            y(angle)=0;
        end    
        for j=1:npop
            dist_y(j)=norm(y(variables) - pop(j,variables));            
        end
        y(obj)=fo(dist_y,neighbors) - (nvar)*y(angle);
        fo_aux=zeros(1,npop);
        for j=1:(npop)
            aux=dist(j,:);
            aux(j)=dist_y(j);
            fo_aux(j)=fo(aux,neighbors) - (nvar)*pop(j,angle);            
        end
        [menor,menorpos]=min(fo_aux);
        if menor < y(obj)
            pop(menorpos,:)=y;
            for j=1:npop
                if j==menorpos
                    dist(menorpos,j)=inf;
                else
                    dist(menorpos,j)=norm(pop(menorpos,variables)-pop(j,variables));
                    dist(j,menorpos)=dist(menorpos,j);
                end
            end
        end
    clc
    fprintf('iteration %u finished', iter);
    iter=iter+1;
    if iter>=itermax
        ok=1;
    end    
	sigma=min(dist); 
	w=(1-t)*a + t*b;
	sigma=w*sigma;
end
for i=1:npop
    for j=(i+1):npop
        dist(i,j)=norm(pop(i,variables)-pop(j,variables));
        dist(j,i)=dist(i,j);
    end
end
for i=1:npop
    pop(i,obj)=fo(dist(:,i),1);
end
resp=pop(:,variables);
end

function resp=fo(vet,k)
aux=sort(vet);
resp=sum(aux(1:k));
end

function resp=gerapop(dim,npop,lb,ub)
crom=lhsdesign(npop,dim);
for i=1:dim
    crom(:,i)=lb(i)+(ub(i)-lb(i))*crom(:,i);
end
resp=crom;
end