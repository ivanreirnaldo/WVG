function [resp]=WVG(nvar,npop,n)
% Weight Vector Generator version 1.0
% nvar -> dimension of the objective space
% npop -> number of weight vecrtors
% n    -> p-norm of the vectors


% Algorithm settings
itermax=6000;               % maximum number of iterations
neighbors=nvar;             % number of neighbors
gap=1e-4;
lb=zeros(1,nvar) + gap;     % lower bound
ub=ones(1,nvar) - gap;      % upper bound
a=1.5;
b=0.1;

% Vector addressing
variables=1:nvar;
obj=nvar+1;

% population start
pop=zeros(npop,obj);
pop(:,variables)=gerapop(nvar,npop,lb,ub);
for i=1:npop
    pop(i,variables)=pop(i,variables)/norm(pop(i,variables),n);
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
    pop(i,obj)=fo(dist(:,i),neighbors);
end
% Start algorithm
ok=0;
iter=1;
while ok==0
    t=iter/itermax;
    w=(1-t)*a + t*b;
    y=zeros(1,obj);
    dist_y=zeros(1,npop);
    i=randi(npop,1);
    delta=normrnd(0,w*sigma(i),1,nvar);
    y(variables)=pop(i,variables) + delta;
    y(variables)=max(y(variables),lb);
    y(variables)=min(y(variables),ub);
    y(variables)=y(variables)/norm(y(variables),n);
    for j=1:npop
        dist_y(j)=norm(y(variables) - pop(j,variables));
    end
    y(obj)=fo(dist_y,neighbors);
    fo_aux=zeros(1,npop);
    for j=1:npop
        aux=dist(j,:);
        aux(j)=dist_y(j);
        fo_aux(j)=fo(aux,neighbors);
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
        sigma=min(dist);
    end
    clc
    fprintf('iteration %u finished', iter);
    if iter>=itermax
        ok=1;
    end
    iter=iter+1;    
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

